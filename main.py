"""
College Event Recommender — terminal version.
Run with: python3 main.py
"""

from recommender import events, get_recommendations


def main():
    user_input = input("Enter your interests (comma-separated): ").strip()

    if not user_input:
        print("No interests entered — please run again and type something.")
        return

    recommendations = get_recommendations(user_input, events, top_n=3)

    if not recommendations:
        print("\nNo events matched those interests — try different or broader keywords.")
        return

    print("\nTop matching events for you:\n")
    for rec in recommendations:
        print(f"{rec['name']} ({rec['category']}) — similarity: {rec['score']:.2f}")
        print(f"  \u2192 {rec['reason']}\n")


if __name__ == "__main__":
    main()
