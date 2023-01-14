'''Convert huggingface datasets to text in the right format'''
import datasets
import random
import json
import re
from tqdm import tqdm

CHUNK_SIZE = 512

def make_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    was_str = False
    if isinstance(lst, str):
        was_str = True
        lst = lst.split(" ")

    for i in range(0, len(lst), n):
        if was_str:
            yield " ".join(lst[i:i + n])
        else:
            yield lst[i:i + n]

def make_line_chunks(context_lines, n):
    '''Chunks but never split within a line'''
    def calc_chunk_length(chunk):
        # return len(tokenizer.tokenize(" ".join(chunk)))
        return len(" ".join(chunk).split(" "))

    if isinstance(context_lines, str):
        context_lines = context_lines.split("\n")

    # Limit individual lines lines to len(n)
    for i in range(len(context_lines)):
        if len(context_lines[i].split(" ")) >= n-1:
            context_lines[i] = " ".join(context_lines[i].split(" ")[:n-1])

    idx = 0
    context_chunks = []
    while idx < len(context_lines):
        context_chunk = []
        answer_chunk = []
        while idx < len(context_lines) and calc_chunk_length(context_chunk + [context_lines[idx]]) < n:
            context_chunk = context_chunk + [context_lines[idx]]
            idx += 1
        context_chunk = "</s>".join(context_chunk)
        context_chunks.append(context_chunk)
    return context_chunks


'''
from sentence_similarity import SentenceSimilarity
def find_most_similar(query, sentences):
    sentence_sim = SentenceSimilarity(sentences)
    return sentence_sim.get_most_similar(query)[0]


# HotpotQA

ds = datasets.load_dataset("ghomasHudson/hotpotExtendedAnoLM")
for split in ["train", "validation"]:
    with open(f"hotpotqa.{split}.txt", 'w') as f:
        for ex in tqdm(ds[split]):
            try:
                # When fact is empty - pick the sentences with smallest cosine similarity
                if split == "train" and "\nFacts: \nAnswer:" in ex["text"]:
                    question = ex["text"].split("Question: ", 1)[1].split("\nContext", 1)[0]
                    context = ex["text"].split("\nContext: ", 1)[1].split("\nFacts:", 1)[0].split(".")
                    new_fact = find_most_similar(question, context).strip()
                    ex["text"] = ex["text"].replace("\nFacts: \nAnswer:", f"\nFacts: {new_fact}\nAnswer:")
            except:
                continue

            f.write(ex["text"].replace("\nAnswer: ", "\n<|answer|> ").replace("Question: ", "<|question|> ").replace("\nFacts: ", "\n<|facts|> ").replace("\nContext: ", "\n<|context|> ").strip())
            f.write(f'\n<|endoftext|>\n\n')
'''




## Style Change
#ds = datasets.load_dataset("ghomasHudson/ao3_style_change")
#for split in ["train", "validation"]:
#    with open(f"style_change.authors.{split}.txt", 'w') as f_out:
#        for ex in tqdm(ds[split]):
#            chunks = list(make_chunks("\n".join(ex["paragraphs"]), CHUNK_SIZE))
#            chunk_idx = 0
#            if ex["authors"] < 3:
#                continue
#            stage_1 = []
#            for chunk in chunks:
#                newline_counter = chunk.count("\n")
#                authors = ex["paragraph-authors"][chunk_idx: chunk_idx + newline_counter+1]
#                chunk_idx += newline_counter
#                assert len(authors) == len(chunk.split("\n"))

#                output = ""
#                prev_a = None
#                num_seen = 0
#                seen = {}
#                for i in range(len(authors)):
#                    if authors[i] not in seen:
#                        output += "<|example|> " + chunk.split("\n")[i].strip() + " <|indexes|> "
#                        seen[authors[i]] = len(seen.keys())
#                    elif authors[i] != prev_a:
#                        output += " <|indexes|> "
#                    prev_a = authors[i]
#                    output += str(seen[authors[i]]) + " "

#                stage_1.append({"text": output, "authors": authors})
#                print(output)
#                # print("<|text|> " + chunk)
#                # print("<|output|> " + " ".join([str(a) for a in authors]) + " | \n")
#                # print("<|endoftext|>\n\n")
#                f_out.write(\
#                        "<|text|> " + chunk +\
#                        "\n<|output|> " + output.replace("  ", " ") + "| " +\
#                        "\n<|endoftext|>\n\n")

#            print()
#            print()

#            # Intermediate level outputs
#            def make_intermediate(stage):
#                new_stage = []
#                for offset in range(0, len(stage)-10, 3):
#                    size = random.randint(5,10)
#                    text = ""
#                    output = ""
#                    authors = []
#                    for i in range(size):
#                        text += stage[i+offset]["text"].strip() + " | "
#                        authors += stage[i+offset]["authors"]
#                    print(len(text.split()))

#                    def get_example(i, text):
#                        '''Get  example corresponding to i'''
#                        #text = text.split("<|text|> ")[1]
#                        examples = text.split("<|example|> ")[1:]
#                        counter = 0
#                        for example in examples:
#                            example, indexes = example.split("<|indexes|> ", 1)
#                            indexes = indexes.split("<|indexes|> ")
#                            for ind in indexes:
#                                ind = ind.split(" | ")[0].strip()
#                                ind = ind.split()
#                                counter += len(ind)
#                            if i < counter:
#                                return example.strip()
#                        print("err")

#                    seen = {}
#                    prev_a = None
#                    for i in range(len(authors)):
#                        if authors[i] not in seen:
#                            output += "<|example|> " + get_example(i, text) + " <|indexes|> "
#                            seen[authors[i]] = len(seen.keys())
#                        elif authors[i] != prev_a:
#                            output += "<|indexes|> "
#                        prev_a = authors[i]
#                        output += str(seen[authors[i]]) + " "
#                    output += "| "
#                    f_out.write(\
#                        "<|text|> " + text.replace("  ", " ") +\
#                        "\n<|output|> " + output.replace("  ", " ") +\
#                        "\n<|endoftext|>\n\n")
#                    new_stage.append({"text": output, "authors": authors})
#                return new_stage

#            stage_2 = make_intermediate(stage_1)
#            make_intermediate(stage_2)
#    '''


#    # Task 1
#    '''
#    with open(f"style_change.multi-author.{split}.txt", 'w') as f_out_multi:
#        with open(f"style_change.changes.{split}.txt", 'w') as f_out_changes:
#            for ex in tqdm(ds[split]):
#                chunks = list(make_chunks("\n".join(ex["paragraphs"]), 50))
#                chunk_idx = 0
#                ex["changes"] = list(range(len(ex["changes"])))
#                for chunk in chunks:
#                    newline_counter = chunk.count("\n")
#                    chunk = chunk.split("\n")
#                    changes = ex["changes"][chunk_idx:chunk_idx + newline_counter]
#                    # assert len(changes) == len(chunk) - 1
#                    chunk_idx += newline_counter
#                    f_out_multi.write(\
#                        "<|text|> " + "\n".join(chunk) +\
#                        "\n<|output|> " + str(sum(changes)!=0) +\
#                        "\n<|endoftext|>\n\n"
#                    )

#                    f_out_changes.write(\
#                        "<|text|> " + "\n".join(chunk) +\
#                        "\n<|changes|> " + " ".join([str(s) for s in changes]) +\
#                        "\n<|endoftext|>\n\n"
#                    )

#                    # # Layer Two - Train model to do "if any true -> true"
#                    num_of_votes = len(chunks)
#                    if random.random() > 0.5:
#                        # True (multiauthor)
#                        proportion_hero = random.random()
#                        num_hero = round(num_of_votes * proportion_hero)
#                        vote_list = [True] * num_hero + [False] * (num_of_votes - num_hero)
#                        random.shuffle(vote_list)
#                    else:
#                        # False (single author)
#                        vote_list = [False] * num_of_votes
#                    f_out_multi.write(f'<|text|> {" | ".join([str(v) for v in vote_list])} | \n')
#                    f_out_multi.write(f'<|output|> {str(sum(vote_list) != 0)} | \n')

#                    f_out_multi.write(f'<|endoftext|>\n\n')
#    '''


# Char ID
# ds = datasets.load_dataset("ghomasHudson/muld", "Character Archetype Classification")
entries = []
# Layer one
line_count = 0
entry = ""
with open("character_id.train.txt", 'w') as f_out:
    for line in tqdm(open("./character_id_train.json")):
        line = json.loads(line)
        character_name, text = line["input"].split("\n", 1)
        text = text.strip()
        text = text.replace("\n", " ")
        text = re.sub('\\s+', ' ', text)
        if line["output"] == "Villain/Antagonist":
            line["output"] = "Antagonist"
        for chunk in make_chunks(text,CHUNK_SIZE-120):
            line_count += 1
            f_out.write(f'<|question|> Is {character_name} a Hero or an Antagonist?\n')
            f_out.write(f'<|text|> {chunk}\n')
            # f_out.write(chunk)
            # f_out.write("\n\n")
            if character_name.lower() in chunk.lower().split():
                f_out.write(f'<|output|> {line["output"]} | \n')
            else:
                f_out.write(f'<|output|>\n')
            f_out.write(f'<|endoftext|>\n\n')
        # entries.append(entry)

            # # Layer Two - Train model to do majority vote
            # for i in tqdm(range(line_count)):
            num_of_votes = random.randint(5, (CHUNK_SIZE-120)/2)
            proportion_hero = random.random()
            num_hero = round(num_of_votes * proportion_hero)
            vote_list = ["Hero"] * num_hero + ["Antagonist"] * (num_of_votes - num_hero)
            random.shuffle(vote_list)
            f_out.write(f'<|question|> Is {character_name} a Hero or an Antagonist?\n')
            f_out.write(f'<|text|> {" | ".join(vote_list)} | \n')
            if num_hero > num_of_votes/2:
                f_out.write(f'<|output|> Hero | \n')
            else:
                f_out.write(f'<|output|> Antagonist | \n')
            f_out.write(f'<|endoftext|>\n\n')

# random.shuffle(entries)
# for entry in entries:
#     f_out.write(entry)

#breakpoint()


'''
# Translation
ds = datasets.load_dataset("ghomasHudson/long_contra_pro")

# Take chunks of CHUNK_SIZE
for split in ds:
    with open(f"long_contra_pro.{split}.txt", 'w') as f_out:
        for ex in ds[split]:
                en_lines = ex["translation"]["en"].split("\n")
                de_lines = ex["translation"]["de"].split("\n")
                en_chunks = list(make_chunks(en_lines, 10))
                de_chunks = list(make_chunks(de_lines, 10))
                assert len(en_chunks) == len(de_chunks)
                en_chunks = ["\n".join(c) for c in en_chunks]
                de_chunks = ["\n".join(c) for c in de_chunks]
                for i in range(len(en_chunks)):
                    f_out.write(f'<|input|> {en_chunks[i]}\n')
                    f_out.write(f'<|output|> {de_chunks[i]}\n')
                    f_out.write(f'<|endoftext|>\n')


'''
'''
# Summarization
ds = datasets.load_dataset("ghomasHudson/booksum_ds", use_auth_token=True)
with open("booksum.txt", 'w') as f_out:
    for ex in tqdm(ds["train"]):
        f_out.write(f'<|input|> {ex["input"]}\n')
        f_out.write(f'<|output|> {ex["output"][0]}\n')
        f_out.write(f'<|endoftext|>\n')
'''
