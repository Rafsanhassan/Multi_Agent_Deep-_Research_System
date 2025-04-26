# 🤖 Multi_Agent_Deep_Research_System

An advanced, modular AI-driven research assistant built with a multi-agent architecture using **LangGraph**, **LangChain**, **Tavily API**, and **Chroma**. The system performs intelligent web research and synthesizes coherent answers to complex queries through iterative reasoning and source evaluation.

---

## 🧠 Overview

The Deep Research AI Agentic System is designed to simulate the cognitive process of a human researcher. It uses multiple AI agents to:
- **Crawl and extract** information from the web
- **Evaluate and store** findings
- **Identify knowledge gaps**
- **Synthesize coherent, well-cited responses**

By separating concerns between agents (e.g., research vs. synthesis), the system ensures clarity, focus, and deep understanding of any research topic.

---

## 🔧 Tech Stack

- **LangGraph** – State-based workflow orchestration  
- **LangChain** – Agent and tool integration  
- **Tavily API** – Online search and information crawling  
- **ChromaDB** – Vector store for semantic retrieval  
- **LLMs** – Language models powering the agents

---

## 🏗️ Architecture

### Agent Roles

- **🔍 Research Agent**
  - Gathers information via Tavily API
  - Evaluates source reliability
  - Identifies knowledge gaps and additional sub-questions

- **📝 Synthesis Agent**
  - Organizes and summarizes collected data
  - Drafts human-readable answers
  - Highlights missing or unclear points

### State Management

All progress, metadata, and collected data are tracked using LangGraph’s state machine via an `AgentState` class which holds:
- Original query
- Current research findings
- Draft answers
- Source documents and citations
- Research flags (e.g., is complete, needs refinement)

---

## 🔄 Workflow

1. **Initial Query** → Broad search by Research Agent  
2. **Store Findings** → Add results to ChromaDB  
3. **Check Gaps** → If any, repeat with focused questions  
4. **Draft Answer** → Synthesized by Synthesis Agent  
5. **Refine** → Repeat if more information is required  
6. **Final Output** → Polished, well-cited response

---


