from retrieve import retrieve
from generate import generate_answer, render_answer
from reviewer_agent import review_answer

def main():
    question = input("Question: ")
    hits = retrieve(question, k=5)
    answer = generate_answer(question, hits)
    print(render_answer(answer, hits))
    if hits and answer:
        print(review_answer(hits, answer))
 
 
if __name__ == "__main__":
    main()
 