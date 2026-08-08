import json
import boto3
import uuid
from datetime import datetime
from decimal import Decimal

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
redshift_data = boto3.client('redshift-data', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
s3 = boto3.client('s3', region_name='us-east-1')

TABLE_NAME = 'conversation-history'
WORKGROUP_NAME = 'default-workgroup'
DATABASE_NAME = 'dev'
BEDROCK_MODEL_ID = 'amazon.nova-micro-v1:0'
DOCUMENTS_BUCKET = 'pavan-cfpb-uploads'
DOCUMENTS_PREFIX = 'results/'

table = dynamodb.Table(TABLE_NAME)


def lambda_handler(event, context):
    # Handle CORS preflight requests before anything else
    http_method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method')
    if http_method == 'OPTIONS':
        return _response(200, {})

    try:
        body = json.loads(event.get('body') or '{}')
        user_message = body.get('message', '').strip()
        session_id = body.get('session_id') or str(uuid.uuid4())

        if not user_message:
            return _response(400, {'error': 'message is required'})

        intent = classify_intent(user_message)

        if intent == 'structured_query':
            answer = handle_structured_query(user_message)
        elif intent == 'document_query':
            answer = handle_document_query(user_message)
        else:
            answer = handle_general_chat(user_message, session_id)

        save_turn(session_id, user_message, answer)

        return _response(200, {
            'session_id': session_id,
            'intent': intent,
            'answer': answer
        })

    except Exception as e:
        return _response(500, {'error': str(e)})


def classify_intent(message):
    prompt = f"""Classify the following user question into exactly one category. Respond with ONLY the category name, nothing else.

Categories:
- structured_query: questions about counts, statistics, trends, comparisons, or numbers from a complaints database (e.g. "how many complaints about X", "top states for Y")
- document_query: questions about a specific uploaded document, letter, or file (e.g. "what does this document say", "summarize the complaint letter")
- general_chat: greetings, general questions, or anything not covered above

User question: "{message}"

Category:"""

    result = invoke_bedrock(prompt, max_tokens=20)
    result = result.strip().lower()

    if 'structured_query' in result:
        return 'structured_query'
    elif 'document_query' in result:
        return 'document_query'
    else:
        return 'general_chat'


SCHEMA_DESCRIPTION = """
Table: complaints
Columns:
- complaint_id (bigint)
- date_received (date)
- product (varchar)
- sub_product (varchar)
- issue (varchar)
- company (varchar)
- state (varchar, 2-letter US state code)
- narrative (varchar)
- company_response (varchar)
- timely_response (varchar)
- narrative_clean (varchar)
- narrative_word_count (integer)

Table: complaint_sentiment
Columns:
- complaint_id (bigint, joins to complaints.complaint_id)
- sentiment (varchar, e.g. 'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'MIXED')
- positive_score (double precision)
- negative_score (double precision)
- neutral_score (double precision)
- mixed_score (double precision)
"""


def handle_structured_query(message):
    sql = generate_sql(message)

    if sql is None:
        return "I couldn't turn that into a valid data query. Could you rephrase your question?"

    try:
        rows, columns = run_redshift_query(sql)
    except Exception as e:
        return f"I generated a query but it failed to run against the database. Error: {str(e)}"

    return summarize_results(message, sql, columns, rows)


def generate_sql(message):
    prompt = f"""You are a SQL expert. Given the schema below, write a single valid Amazon Redshift SQL query (PostgreSQL-compatible syntax) that answers the user's question.

Schema:
{SCHEMA_DESCRIPTION}

Rules:
- Return ONLY the SQL query, no explanation, no markdown code fences, no semicolon at the end.
- Always add a LIMIT 50 unless the question asks for a single aggregate number (like a COUNT or AVG).
- Use ILIKE for case-insensitive text matching on issue/product/company columns.
- Join complaints and complaint_sentiment on complaint_id when the question involves sentiment.

User question: "{message}"

SQL query:"""

    result = invoke_bedrock(prompt, max_tokens=300).strip()
    result = result.replace('```sql', '').replace('```', '').strip()

    if not result.upper().startswith('SELECT'):
        return None

    forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT']
    if any(word in result.upper() for word in forbidden):
        return None

    return result


def run_redshift_query(sql):
    exec_response = redshift_data.execute_statement(
        WorkgroupName=WORKGROUP_NAME,
        Database=DATABASE_NAME,
        Sql=sql
    )
    query_id = exec_response['Id']

    import time
    for _ in range(30):
        status_response = redshift_data.describe_statement(Id=query_id)
        status = status_response['Status']
        if status == 'FINISHED':
            break
        elif status in ('FAILED', 'ABORTED'):
            raise Exception(status_response.get('Error', 'Query failed'))
        time.sleep(1)
    else:
        raise Exception('Query timed out')

    result = redshift_data.get_statement_result(Id=query_id)
    columns = [col['name'] for col in result['ColumnMetadata']]

    rows = []
    for record in result['Records']:
        row = []
        for field in record:
            value = list(field.values())[0] if field else None
            row.append(value)
        rows.append(row)

    return rows, columns


def summarize_results(message, sql, columns, rows):
    preview_rows = rows[:20]
    rows_str = "\n".join([str(dict(zip(columns, r))) for r in preview_rows])

    prompt = f"""You are a data analyst assistant. The user asked: "{message}"

This SQL query was run:
{sql}

Results ({len(rows)} row(s) returned, showing up to 20):
{rows_str}

Write a brief, natural-language answer to the user's question based on these results. 
Include specific numbers where relevant. Do not mention SQL or databases explicitly, just answer naturally."""

    return invoke_bedrock(prompt, max_tokens=400)


def handle_document_query(message):
    documents = load_all_documents()

    if not documents:
        return "There aren't any processed documents available yet. Upload a document through the dashboard's document intelligence feature first, and I'll be able to answer questions about it."

    relevant_docs = find_relevant_documents(message, documents)

    if not relevant_docs:
        return "I couldn't find a document matching that question. There's currently 1 processed document available — try asking about it more generally, e.g. 'what is the complaint letter about?'"

    return answer_from_documents(message, relevant_docs)


def load_all_documents():
    documents = []
    try:
        response = s3.list_objects_v2(Bucket=DOCUMENTS_BUCKET, Prefix=DOCUMENTS_PREFIX)
        for obj in response.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.json'):
                continue
            file_obj = s3.get_object(Bucket=DOCUMENTS_BUCKET, Key=key)
            content = json.loads(file_obj['Body'].read())
            documents.append(content)
    except Exception:
        pass
    return documents


def find_relevant_documents(message, documents):
    if len(documents) <= 5:
        return documents

    message_words = set(message.lower().split())
    scored = []
    for doc in documents:
        text = (doc.get('extracted_text', '') + ' ' + doc.get('summary', '')).lower()
        score = sum(1 for word in message_words if word in text)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:3] if score > 0] or [documents[0]]


def answer_from_documents(message, documents):
    docs_str = ""
    for i, doc in enumerate(documents, 1):
        docs_str += f"""
Document {i}: {doc.get('filename', 'unknown')}
Summary: {doc.get('summary', 'N/A')}
Sentiment: {doc.get('sentiment', 'N/A')}
Full text: {doc.get('extracted_text', 'N/A')}
"""

    prompt = f"""You are a helpful assistant for a consumer complaints analytics platform.
The user asked about an uploaded document: "{message}"

Here is the available document data:
{docs_str}

Answer the user's question based on this document content. Be specific and reference 
details from the document where relevant. If the question can't be answered from 
this content, say so honestly."""

    return invoke_bedrock(prompt, max_tokens=400)


def handle_general_chat(message, session_id):
    history = get_recent_history(session_id)
    context_str = ""
    if history:
        context_str = "\n".join([f"User: {h['user_message']}\nAssistant: {h['answer']}" for h in history])

    prompt = f"""You are a helpful, concise AI assistant for a consumer complaints analytics platform.

Recent conversation:
{context_str}

User: {message}

Respond naturally and helpfully."""
    return invoke_bedrock(prompt, max_tokens=400)


def invoke_bedrock(prompt, max_tokens=300):
    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.3}
        })
    )
    result = json.loads(response['body'].read())
    return result['output']['message']['content'][0]['text']


def save_turn(session_id, user_message, answer):
    table.put_item(Item={
        'session_id': session_id,
        'timestamp': datetime.utcnow().isoformat(),
        'user_message': user_message,
        'answer': answer
    })


def get_recent_history(session_id, limit=5):
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(session_id),
        ScanIndexForward=False,
        Limit=limit
    )
    items = response.get('Items', [])
    return list(reversed(items))


def _response(status_code, body_dict):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'POST, OPTIONS'
        },
        'body': json.dumps(body_dict, default=str)
    }
