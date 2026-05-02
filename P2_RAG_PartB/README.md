# Project Overview

## My Contribution

### Multi-Level Retrieval System
Implemented a multi-level retrieval pipeline across four key dimensions:

- **General Financial Performance**
- **Specialized Financial Metrics**
- **Market Sentiment & Risk**
- **Multi-Query Integration Queries** (as outlined in Appendix B)

This system enables more comprehensive and context-aware financial analysis by combining multiple perspectives during retrieval.

---

### HPC Environment Setup

Configured the high-performance computing (HPC) environment to support scalable experimentation:

- Set up **SLURM job arrays**  
- Enabled **parallel execution of all 5 agents**  
- Optimized batching for the **5-model evaluation grid** on the supercomputer  

This significantly improves computational efficiency and reduces total runtime.

---

## ChromaDB Integration

### Setup Process

To connect and use ChromaDB:

1. Obtained the project folder from the contributor responsible for **RAG Part 1**  
2. Executed the code provided in **Cell 2 of the Colab notebook**  

This initializes the database and allows access to the pre-built vector store.

---

## Notes

- Ensure all dependencies are installed before running the notebook  
- Verify correct file paths when loading the RAG folder  