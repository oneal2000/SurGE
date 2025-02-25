import re
from markdownParser import *
import numpy as np
from scipy.spatial.distance import pdist

import gc

def eval_coverage(target_cites:list, gen_cite_map:dict):
    gen_cites = []
    for k,v in gen_cite_map.items():
        gen_cites.append(v)
        
    all = len(target_cites)
    hit = 0
    
    for c in target_cites:
        if c in gen_cites:
            hit += 1
            
    return hit/all

def compute_cd(embeddings, metric="euclidean"):
    embeddings = np.array(embeddings)  
    distances = pdist(embeddings, metric=metric)  
    return np.mean(distances)

def eval_diversity(gen_cite_text:list, gt_bertopic, model):
    topic, prob = model.transform(gen_cite_text)
    cd = compute_cd(prob)
    return cd/gt_bertopic


def eval_relevance_paper(target_survey,gen_cite_map:dict,cite_content:dict,nli_model):
    if len(gen_cite_map) == 0:
        return 0
    
    all = len(gen_cite_map)
    hit = 0
    nli_pairs = []
    for k,v in gen_cite_map.items():
        if v in target_survey['all_cites']:
            hit += 1
        else:
            if cite_content[k][0] != "[NOTEXIST]":
                nli_pairs.append(cite_content[k])
    
    if len(nli_pairs) > 0:  
        print("Eval relevanve paper")
        print(gen_cite_map)
        print(nli_pairs)
        
        scores = nli_model.predict(nli_pairs)
        
        label_mapping = ['contradiction', 'entailment', 'neutral']
        # 计算 entailment 的 Softmax 值
        # values = torch.nn.functional.softmax(torch.tensor(scores), dim=1)
        # entailment_probs = values[:,1].tolist()
        # print("Papaer possibilities")
        # print(entailment_probs)
        # for i in entailment_probs:
        #     if i > 0.6:
        #         hit += 1
        
        for c,e,n in scores:
            if e > c and e > n:
                hit += 1
            elif n > c and n > e:
                if e > c:
                    hit += 0.5
        
    return hit/all

def eval_relevance_section(nli_pairs_origin,nli_model):
    if len(nli_pairs_origin) == 0:
        return 0
    
    missed = 0
    nli_pairs = []
    for citation, subtitle in nli_pairs_origin:
        if citation != "[NOTEXIST]":
            nli_pairs.append((subtitle,citation))
        else:
            missed += 1
    #print("Eval relevanve section")
    #print(nli_pairs)
    
    
    all = len(nli_pairs) + missed
    hit = 0
    if len(nli_pairs) > 0:
        scores = nli_model.predict(nli_pairs)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        
        
        for c,e,n in scores:
            if e > c and e > n:
                hit += 1
            elif n > c and n > e:
                if e > c:
                    hit += 0.5
        
        # values = torch.nn.functional.softmax(torch.tensor(scores), dim=1)
        # entailment_probs = values[:,1].tolist()
        #print("section possibilities")
        #print(entailment_probs)
        # for i in entailment_probs:
        #     if i > 0.6:
        #         hit += 1
        
    return hit/all

def eval_relevance_sentence(nli_pairs_origin,nli_model):
    if len(nli_pairs_origin) == 0:
        return 0
    
    missed = 0
    nli_pairs = []
    
    for citation, sentence in nli_pairs_origin:
        if "[NOTEXIST]" != citation:
            nli_pairs.append((sentence,citation))        
        else: 
            missed += 1
            
    #print("Eval relevanve sentence")
    #print(nli_pairs)
    
    all = len(nli_pairs) + missed
    hit = 0
    if len(nli_pairs) > 0:
        scores = nli_model.predict(nli_pairs)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        # 计算 entailment 的 Softmax 值
        # values = torch.nn.functional.softmax(torch.tensor(scores), dim=1)
        # entailment_probs = values[:,1].tolist()
        # print("sentence possibilities")
        # print(entailment_probs)
        # for i in entailment_probs:
        #     if i > 0.6:
        #         hit += 1
        
        for c,e,n in scores:
            if e > c and e > n:
                hit += 1
            elif n > c and n > e:
                if e > c:
                    hit += 0.5
        
    return hit/all
    
    
def extract_references(text):
    pattern = r'\[(\d+)\]'
    matches = re.finditer(pattern, text)
    results = []
    
    for match in matches:
        ref_number = int(match.group(1))
        start_pos = match.start()
        sentence_start = text.rfind('.', 0, start_pos) + 1
        sentence_end = text.find('.', start_pos)
        if sentence_end == -1:
            sentence_end = len(text)
        sentence = text[sentence_start:sentence_end].strip()
        results.append((ref_number, sentence))
    
    return results

def extract_cites_with_subtitle_and_sentence(psg_node:MarkdownNode):
    res = []
    text = "\n".join(psg_node.content)
    tmp_res = extract_references(text)
    if "root" not in psg_node.title and "Abstract:" not in psg_node.title :
        for ref_number, sentence in tmp_res:
            res.append((ref_number, psg_node.title, sentence))
    
    for child in psg_node.children:
        res.extend(extract_cites_with_subtitle_and_sentence(child))
    
    return res


def get_content_list(psg_node:MarkdownNode):
    res = []
    text = "\n".join(psg_node.content)
    if len(text) >= 100:
        res.append(text)
    for child in psg_node.children:
        tmp_res = get_content_list(child)
        res.extend(tmp_res)
        
    return res

def get_logic_check_prompt(sentence):
    prompt = evaluation_prompt = f"""You are an advanced AI language evaluator. Your task is to assess the logical coherence and clarity of text based on the following criteria:

1. **Fluency & Coherence** – Does the text flow naturally? Are the sentences well-connected and easy to read?
2. **Logical Clarity** – Is the reasoning clear and structured? Does the argument progress logically without contradictions?
3. **Avoidance of Redundancy** – Does the text avoid unnecessary repetition?
4. **Clarity of Description** – Are ideas, concepts, or events described in a way that is easy to understand?
5. **Absence of Errors** – Does the text contain grammatical mistakes, spelling errors, or factual inconsistencies?

You will provide a **score from 0 to 5** based on these criteria, along with no explanation.

### **Scoring Guide**

**5 – Excellent**
- The text is highly fluent, with smooth transitions and a natural flow.
- The logical progression is clear, well-structured, and easy to follow.
- There is no redundancy; each sentence contributes meaningfully.
- Descriptions are precise and unambiguous.
- No spelling, grammatical, or factual errors.

**Example (5/5):**
"The rise of artificial intelligence has transformed industries worldwide. From healthcare to finance, AI-driven innovations have streamlined processes and improved efficiency. As companies integrate AI technologies, they must also address ethical concerns to ensure responsible use."

---

**4 – Good**
- The text is mostly fluent, with minor awkward transitions.
- Logical progression is clear but may contain slight inconsistencies.
- Some minor redundancy or repetition.
- Descriptions are mostly clear, with minor ambiguities.
- Very few spelling or grammatical errors.

**Example (4/5):**
"Artificial intelligence is changing industries such as healthcare and finance. AI helps improve efficiency, and many businesses are adopting AI. However, ethical concerns remain, and companies must ensure responsible AI use."

---

**3 – Average**
- The text is understandable but has noticeable awkward phrasing.
- Logical flow is inconsistent; some points may feel out of place.
- Some redundancy or repetition that slightly affects readability.
- Certain descriptions are vague or unclear.
- Contains some spelling or grammatical mistakes but remains readable.

**Example (3/5):**
"Artificial intelligence has many applications. In healthcare, finance, and many other sectors, AI helps a lot. Companies are using AI more and more. But AI has risks, and companies need to consider them. AI can be useful if used correctly."

---

**2 – Poor**
- The text is difficult to read due to awkward structure and poor fluency.
- Logical inconsistencies make the argument unclear.
- Repetitive phrases make the content tedious.
- Descriptions are vague, making it hard to understand key points.
- Multiple grammatical and spelling errors.

**Example (2/5):**
"AI is being used in many places. AI is in healthcare. AI is in finance. AI is also used in business. Many people use AI, and AI is useful. But AI has risks. AI must be used in the right way. AI is good, but it can be bad."

---

**1 – Very Poor**
- The text is highly disjointed, making it hard to read.
- Logical flow is almost nonexistent, with abrupt topic shifts.
- Redundant sentences add no value.
- Descriptions are confusing or overly vague.
- Frequent spelling and grammatical mistakes.

**Example (1/5):**
"AI is good. AI is everywhere. In finance. In hospitals. People use AI. Many companies AI. But AI risk. AI must be safe. AI can help. AI problem big. AI future."

---

**0 – Incoherent**
- The text is completely nonsensical or unreadable.
- No logical progression or coherence.
- Extreme redundancy or word salad.
- Severe errors make it impossible to understand.

**Example (0/5):**
"AI good. Many AI. Business AI. Finance hospital AI. More AI need. AI problem bad. Future AI better. AI help. Risk AI."

---
### **Instruction**  
Now evaluate the following paragraph based on the criteria above and provide a score from 0 to 5.  

### **Paragraph:**  
{sentence}  

### **Output Format:**  
Provide only the score as a single number (0-5), without any additional explanation or comments.
"""
    return prompt


def chat_openai(prompt, client,try_number):
    if try_number == 5:
        print("Failed to get valid response after 5 tries.")
        #result_queue.put(None)
    #print(f"Try {try_number} time")
    #print("Query:")
    #print(prompt)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            max_tokens=100
        )
        #print(f"Answer:{response.choices[0].message.content}")
        
        ans = None
        if not re.match(r'^[0-5]$', response.choices[0].message.content):
            ans =  chat_openai(prompt,client,try_number + 1)
        else :
            ans = int(response.choices[0].message.content)
        #result_queue.put(ans)
    except Exception as e:
        print(f"An error occurred: {e}")
        return chat_openai(prompt,client,try_number + 1)
        
    return ans



    
def eval_logic_client(psg_node: MarkdownNode, client):
    psgs = get_content_list(psg_node)
    np.random.seed(42)
    num_samples = min(len(psgs), 20)
    psgs = np.random.choice(psgs, num_samples, replace=False)

    result = 0
    
    for i in range(len(psgs)):
        if len(psgs[i]) > 1000:
            psgs[i] = psgs[i][:1000]
            if psgs[i][-1] != '.':
                psgs[i] = psgs[i][:psgs[i].rfind('.')]
        result += chat_openai(get_logic_check_prompt(psgs[i]), client, 0)

        
    return result/len(psgs)


# def chat(text,model,tokenizer,try_number):
#     if try_number == 5:
#         print("Failed to get valid response after 5 tries.")
#         return None
    
#     prompt = text
#     messages = [
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     # print("TOKENLEN",len(model_inputs['input_ids'][0]))

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=16
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
#     #print(response)

#     response = response.strip('.')
    
#     ans = None
    
#     if not re.match(r'^[0-5]$', response):
#         ans =  chat(text,model,tokenizer,try_number + 1)
#     else :
#         ans = int(response)
    
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     return ans
    

# def eval_logic(psg_node: MarkdownNode, model, tokenizer):
#     psgs = get_content_list(psg_node)
#     np.random.seed(42)
#     psgs = np.random.choice(psgs, 20, replace=False)
    
#     result = 0
#     for i in range(len(psgs)):
#         if len(psgs[i]) > 1000:
#             psgs[i] = psgs[i][:1000]
#             if psgs[i][-1] != '.':
#                 psgs[i] = psgs[i][:psgs[i].rfind('.')]
                
#         prompt = get_logic_check_prompt(psgs[i])
#         result += chat(prompt,model,tokenizer,0)
    
#     return result/len(psgs)
        
      
    