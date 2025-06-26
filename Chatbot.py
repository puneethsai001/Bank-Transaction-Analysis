import pandas as pd
import google.generativeai as genai

df = pd.read_csv("Complete_Transaction_Statement.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Gemini API configuration
genai.configure(api_key="AIzaSyCJWY-UtX-6yHluoJAZglXMof3AzvjwZNQ")
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

monthly_spend = (
    df[df['Debit'].notnull()]
    .groupby([df['Date'].dt.to_period('M'), 'Category'])['Debit']
    .sum()
    .unstack()
    .fillna(0)
)
all_txns = df[['Date', 'Description', 'Credit', 'Debit', 'Category']].to_string(index=False)

context = (
    "You are a helpful AI finance assistant. Use the user's recent transactions and category-wise monthly spending to provide financial insights.\n\n"
    f"All Transactions:\n{all_txns}\n\n"
    f"Monthly Spending by Category (AED):\n{monthly_spend.to_string()}\n\n"
)

print("Finance Chatbot started. Ask questions like:\n"
      "- How much did I spend on shopping last month?\n"
      "- Suggest ways I can save more money.\n"
      "- Which category do I overspend in?\n"
      "Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Stay smart with your money. Goodbye!")
        break
    try:
        prompt = context + f"User: {user_input}\nAssistant:"
        response = model.generate_content(prompt)
        print("Chatbot:", response.text.strip())
    except Exception as e:
        print("Error:", e)
