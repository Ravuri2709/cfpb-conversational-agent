# CFPB Conversational AI Agent

A conversational AI system that answers natural-language questions about consumer complaint data by generating and executing SQL queries in real time — built on AWS Bedrock, Redshift, Lambda, and DynamoDB.

**Live demo:** https://dkd0nbsq74aem.cloudfront.net/chat.html
**API endpoint:** `https://5on9xrn00l.execute-api.us-east-1.amazonaws.com/default/cfpb-agent-router`

This project extends [cfpb-aws-data-pipeline](https://github.com/Ravuri2709/cfpb-aws-data-pipeline), reusing its Redshift data warehouse and adding a new conversational orchestration layer on top.

## What it does

Ask a question in plain English — "What are the top 5 most common complaint issues?" — and the agent:

1. Classifies the intent (structured data question vs. document question vs. general chat)
2. Generates a SQL query against the complaints database using Amazon Bedrock
3. Executes it live against Amazon Redshift Serverless
4. Summarizes the real results back into a natural-language answer
5. Persists conversation history in DynamoDB for multi-turn context

For document-related questions, the agent retrieves relevant processed documents (extracted via Amazon Textract) from S3 and answers using their actual content.

Every answer sourced from live data is visually flagged in the UI so it's clear when the agent is reasoning from real numbers versus general conversation.

## Architecture
