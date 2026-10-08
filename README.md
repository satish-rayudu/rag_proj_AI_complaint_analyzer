# 🤖 AI Complaint Analyzer – Domain-Independent RAG System

A domain-independent **RAG (Retrieval-Augmented Generation) based Customer Complaint Analysis System** that uses semantic retrieval to identify relevant complaint policies and analyze customer complaints across multiple domains.

## 📌 Project Overview

The system analyzes customer complaints from different domains and provides:

- Domain
- Complaint Category
- Complaint Summary
- Issue Identified
- Severity
- Priority
- Relevant Policy
- Recommended Action

## 🧠 RAG Architecture

```text
Customer Complaint
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Vector Database
        ↓
Semantic Similarity Retrieval
        ↓
Relevant Domain Policy
        ↓
Complaint Analysis
        ↓
Domain + Category + Severity + Priority
        ↓
Recommended Action
