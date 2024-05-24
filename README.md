pip install -r requirements.txt



Current set up
run the Simple RAG overview python notebook

run the check_relevance_score.py - checks how relevant the 
documents from the db is to a given query

run query_data2.py - simple RAG for Q&A 

Things to look into to improve RAG model

change database creation = make each token/vector's base text smaller?

change LLM model (currently mistral)

change embedding model (currently sentence-transformers/all-MiniLM-L6-v2 from huggingfact)