# Generalisable_abuse
A repo to try to generalise abusive language detection

## Structure.

All data resides in the data folder. And all code in the src folder.

Further, the source folder is subdivided into different folders, where the top-level folder only contains an ```init.py``` file and a single file to run all models (which is run with options selecting dataset, model, parameters, etc.) while each individual model is contained within its own folder, and finally there is a folder containing shared resources such as feature generation scripts, data reading, etc.

## Features

Represent sentences as:

- LIWC
- Word length
- Word complexity
- Sentiment
- POS (PTB)

## Datasets

- Use Davidson, Waseem, Waseem & Hovy to show adaptability.
