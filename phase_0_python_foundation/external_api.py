
import requests


def get_todo(todo_id: int) -> dict:
    url = f"https://jsonplaceholder.typicode.com/todos/{todo_id}"

    try:
        response = requests.get(url, timeout=10)

        # Raise an error if the API returns an unsuccessful status
        response.raise_for_status()

        # Convert the JSON response into a Python dictionary
        todo = response.json()

        return todo

    except requests.exceptions.Timeout:
        print("The API request timed out.")

    except requests.exceptions.HTTPError as error:
        print("HTTP error:", error)

    except requests.exceptions.RequestException as error:
        print("Request failed:", error)

    return {}


def main():
    todo_id = 10
    todo = get_todo(todo_id)

    if todo:
        print("\n--- TODO DETAILS ---")
        print("ID:", todo["id"])
        print("Title:", todo["title"])
        print("Completed:", todo["completed"])
        print("User ID:", todo["userId"])
    else:
        print("Could not retrieve the todo.")


if __name__ == "__main__":
    main()
