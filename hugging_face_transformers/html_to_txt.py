# %%
from bs4 import BeautifulSoup
import pandas as pd
import re

# %%
soup = BeautifulSoup(open("data/dr-pepper.html"), "html.parser")
# %%
# %%
initial_list = [re.sub("[^A-Za-z0-9]+", " ", x.text) for x in soup.find_all("span")]
# %%
cleaned_list = [x for x in initial_list if len(x) > 100]
# %%
df = pd.DataFrame(
    {"sentence": cleaned_list, "length": [len(x) for x in cleaned_list]}
).drop_duplicates()
# %%
df.to_csv("data/dr-pepper.csv", index=False)