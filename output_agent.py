import json
from datetime import datetime
from assignment_agent import assign_priority_and_dates
from extractor_agent import read_transcript, extract_action_items


def save_to_json(final_items, filename="action_items_output.json"):
    """Saves the final action items into a JSON file"""
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(final_items, file, indent=2)
    print(f"Results saved to {filename}")


def print_as_table(final_items):
    """Prints the action items in a clean table format"""
    print("\n" + "=" * 90)
    print(f"{'#':<3}{'Task':<40}{'Assignee':<12}{'Deadline':<14}{'Priority':<10}")
    print("=" * 90)

    for i, item in enumerate(final_items, 1):
        task = item['task'][:37] + "..." if len(item['task']) > 37 else item['task']
        print(f"{i:<3}{task:<40}{item['assignee']:<12}{item['deadline']:<14}{item['priority']:<10}")

    print("=" * 90)


if __name__ == "__main__":
    transcript = read_transcript("sample_transcript.txt")
    print("Transcript loaded. Extracting action items...\n")

    raw_items = extract_action_items(transcript)
    print(f"{len(raw_items)} action items found. Assigning priority and dates...\n")

    final_items = assign_priority_and_dates(raw_items)

    print_as_table(final_items)
    save_to_json(final_items)