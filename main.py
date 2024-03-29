import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from nltk.tokenize import sent_tokenize
import random
import nltk

nltk.download('punkt')

# Load pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to preprocess the dataset
def preprocess_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    return sentences

# Function to add variations to sentences
def add_variations(sentence):
    # Split sentence into tokens
    tokens = tokenizer.tokenize(sentence)
    # Introduce random variations
    for i, token in enumerate(tokens):
        if random.random() < 0.2:  # 20% chance of introducing variation
            # Randomly replace token with synonym
            synonyms = get_synonyms(token)
            if synonyms:
                tokens[i] = random.choice(synonyms)
    # Reconstruct sentence from tokens
    return tokenizer.convert_tokens_to_string(tokens)

# Function to get synonyms of a word using NLTK WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in nltk.corpus.wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Custom dataset for fine-tuning GPT-2
class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, sentences, max_length=512):
        self.examples = []
        for sentence in sentences:
            # Add variations to each sentence
            for _ in range(5):  # Generate 5 variations per sentence
                variation = add_variations(sentence)
                self.examples.append(tokenizer.encode(variation, max_length=max_length, truncation=True))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])

# Fine-tune GPT-2 model on custom dataset
def fine_tune_gpt2(sentences, model):
    # Preprocess dataset
    dataset = CustomTextDataset(tokenizer, sentences)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir="./gpt2_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()

# Generate text using fine-tuned GPT-2 model
def generate_text(prompt, model, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Main function
def main():
    # Load pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    # Load and preprocess dataset
    sentences = preprocess_dataset("story_corpus.txt")
    # Fine-tune GPT-2 model on custom dataset
    fine_tune_gpt2(sentences, model)
    # Generate text using fine-tuned model
    prompt = "Once upon a time"
    generated_text = generate_text(prompt, model)
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
