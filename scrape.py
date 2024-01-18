from bs4 import BeautifulSoup
import requests
import json

url = "https://wob.coppermind.net/events/515-tress-spoiler-stream/"

# Make a request to the website
response = requests.get(url)

# Parse the HTML.
soup = BeautifulSoup(response.text, 'html.parser')


# Find all divs that contains the Q&A content
entry_contents = soup.find_all('div', class_='entry-content')

# Initialize the Q&A list
qa_list = []
# Loop through each entry_content
for entry_content in entry_contents:
    # Find all h4 (speakers) and p (messages) elements
    speakers = entry_content.find_all('h4', class_='entry-speaker')
    messages = entry_content.find_all('p')

    # Initialize question and answer variables
    question = None
    answer = None

    # Loop through all speakers and their corresponding messages
    for speaker, message in zip(speakers, messages):
        # If speaker is Brandon Sanderson, then the message is an answer
        if speaker.text.strip() == "Brandon Sanderson":
            answer = message.text.strip()
        else: # Otherwise, the message is a question
            question = message.text.strip()

    # Append the Q&A pair to the list
    if question and answer:
        qa_list.append({"Question": question, "Answer": answer})

# Convert the Q&A list to JSON
qa_json = json.dumps(qa_list, indent=4)

# Print the Q&A JSON
print(qa_json)

