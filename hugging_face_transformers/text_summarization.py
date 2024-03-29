# %%[markdown]
# Description: Text summarization using BART

# %%
# Import libraries
from transformers import AutoTokenizer, BartForConditionalGeneration

# %%
# Load pre-trained model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# %%
# Define article to summarize
ARTICLE_TO_SUMMARIZE = """PG&E stated it scheduled the blackouts in response to
forecasts for high winds amid dry conditions. The aim is to reduce the
risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by
the shutoffs which were expected to last through at least midday tomorrow."""
print("ARTICLE TO SUMMARIZE:", ARTICLE_TO_SUMMARIZE)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# %%
# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=30)
summary_text = tokenizer.batch_decode(
    summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
# %%
print("SUMMARY OF ARTICLE:", summary_text)
# %%
