from retrieve import retrieve
from generate import generate_answer
from reviewer_agent import review_answer

question = input("Question: ")
hits = retrieve(question, k=5)
answer = generate_answer(question, hits)
review = review_answer(hits, answer)
print(answer)
print(review)