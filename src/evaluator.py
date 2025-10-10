import json
import argparse
import time
import re
import markdownParser,rougeBleuFuncs,structureFuncs,informationFuncs
import os
from sentence_transformers import CrossEncoder
# from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from FlagEmbedding import FlagModel
from openai import OpenAI
import httpx

def normalize_string(s):
        letters = re.findall(r'[a-zA-Z]', s)
        return ''.join(letters).lower()

class SurGEvaluator:
    def __init__(self,device:str = None,survey_path:str = None,corpus_path:str = None,flag_model_path:str = None, judge_model_path:str = None, bertopic_model_path:str = None,bertopic_embedding_model_path:str = None, nli_model_path:str = None, using_openai:bool = True, api_key:str = None):
        import os
        if device != None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        
        self.corpus_dir = corpus_path

        self.using_openai = using_openai
        if using_openai == True:
            assert api_key != None
            #self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
        
        surveys = []
        self.survey_map = {}
        with open(survey_path,'r',encoding='utf-8') as f:
            surveys = json.load(f)
        for s in surveys:
            self.survey_map[int(s['survey_id'])] = s.copy()
        
        corpus = []
        self.corpus_map = {}
        self.title2docid = {}
        with open(corpus_path,'r',encoding='utf-8') as f:
            corpus = json.load(f)
        for c in corpus:
            self.corpus_map[int(c['doc_id'])] = c.copy()
            self.title2docid[normalize_string(c['Title'])] = int(c['doc_id'])
            
        if flag_model_path == None :
            self.flag_model_path = 'BAAI/bge-large-en-v1.5'
        else:
            self.flag_model_path = flag_model_path
            
        # if judge_model_path == None :
        #     self.judge_model_path = None
        # else:
        #     self.judge_model_path = judge_model_path
            
        # self.judge_model = None
        self.flag_model = None
        # if self.judge_model_path != None:
        #     self.judge_model_tokenizer = AutoTokenizer.from_pretrained(self.judge_model_path)
        # else:
        #     self.judge_model_tokenizer = None    
            
        if nli_model_path == None:
            self.nli_model_path = 'cross-encoder/nli-deberta-v3-base'
        else:
            self.nli_model_path = nli_model_path
            
        self.nli_model = None
            
    def single_eval(self,survey_id,passage_path,eval_list):
        psg_node = markdownParser.parse_markdown(passage_path)
        refs  = markdownParser.parse_refs(passage_path)
        refid2docid = {}

        for refid,ref_title in refs.items():

            if normalize_string(ref_title) in self.title2docid:
                ref_docid = self.title2docid[normalize_string(ref_title)]
                refid2docid[refid] = ref_docid
            else:
                refid2docid[refid] = ref_title
        # print(refid2docid)
        eval_result = {
            "Information_Collection": {
                "Comprehensiveness": {
                    "Coverage": None,
                },
                "Relevance": {
                    "Paper_Level": None,
                    "Section_Level": None,
                    "Sentence_Level": None,
                }
            },
            "Survey_Structure": {
                "Structure_Quality(LLM_as_judge)": None,
                "SH-Recall": None
            },
            "Survey_Content": {
                "Relevance": {
                        "ROUGE-1": None,
                        "ROUGE-2": None,
                        "ROUGE-L": None,
                        "BLEU": None,
                    },
                "Logic": None
            }
        }
        
        if "ROUGE-BLEU" in eval_list or "ALL" in eval_list:
            r1,r2,rl,bleu = rougeBleuFuncs.eval_rougeBleu(self.survey_map[survey_id],psg_node)
            eval_result["Survey_Content"]["Relevance"]["ROUGE-1"] = r1
            eval_result["Survey_Content"]["Relevance"]["ROUGE-2"] = r2
            eval_result["Survey_Content"]["Relevance"]["ROUGE-L"] = rl
            eval_result["Survey_Content"]["Relevance"]["BLEU"] = bleu
        
        if "SH-Recall" in eval_list or "ALL" in eval_list:
            if self.flag_model == None:
                self.flag_model = FlagModel(self.flag_model_path, 
                    query_instruction_for_retrieval="Generate a representation for this title to calculate the similarity between titles:",
                        use_fp16=True)  
            
            sh_recall = structureFuncs.eval_SHRecall(self.survey_map[survey_id],psg_node,self.flag_model)
            eval_result["Survey_Structure"]["SH-Recall"] = float(sh_recall)
        
        if "Structure_Quality" in eval_list or "ALL" in eval_list:
            # if self.judge_model == None and self.using_openai == False:
            #     self.judge_model = AutoModelForCausalLM.from_pretrained(
            #         self.judge_model_path,
            #         torch_dtype= torch.float16,
            #         device_map="auto"
            #     )
            if self.using_openai == True:
                struct_quality = structureFuncs.eval_structure_quality_client(self.survey_map[survey_id],psg_node,self.client)
            else:
                pass 
                # struct_quality = structureFuncs.eval_structure_quality(self.survey_map[survey_id],psg_node,self.judge_model,self.judge_model_tokenizer)
            eval_result["Survey_Structure"]["Structure_Quality(LLM_as_judge)"] = struct_quality 
        
        if "Coverage" in eval_list or "ALL" in eval_list:
            coverage = informationFuncs.eval_coverage(self.survey_map[survey_id]['all_cites'],refid2docid)
            eval_result["Information_Collection"]["Comprehensiveness"]["Coverage"] = coverage
            
        if "Relevance-Paper" in eval_list or "ALL" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            refcontent = {}
            for k,v in refid2docid.items():
                sen_1 = None
                sen_paper = None 
                if isinstance(v,int):
                    tmp_1 = self.corpus_map[v]['Title']
                    tmp_2 = self.corpus_map[v]['Abstract']
                    tmp_title = self.survey_map[survey_id]['survey_title']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: '{tmp_2}'"
                    sen_paper = f"The paper titled '{tmp_1}' with the given abstract could be cited in the paper: '{tmp_title}'."
                    refcontent[k] = (sen_1,sen_paper)
                else:
                    tmp_title = self.survey_map[survey_id]['survey_title']
                    # sen_1 = refcontent[k] = f"There is a paper. Title: '{v}'. The title '{v}' describes the content of the paper."
                    # sen_paper = f"The paper titled '{v}' could be cited in the paper: '{tmp_title}'."
                    sen_1 = "[NOTEXIST]"
                    sen_paper = "[NOTEXIST]"
                    refcontent[k] = (sen_1,sen_paper)
            paper_relevance = None
            if len(refid2docid) > 0:
                paper_relevance = informationFuncs.eval_relevance_paper(self.survey_map[survey_id],refid2docid,refcontent,self.nli_model)
            else:
                paper_relevance = 0
            eval_result["Information_Collection"]["Relevance"]["Paper_Level"] = paper_relevance
        
        if ("Relevance-Section" in eval_list and "Relevance-Sentence" in eval_list) or "ALL" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
            nli_pairs_subtitle = []
            nli_pairs_sentence = []
            for ref_num,subtitle,sentence in extracted_cites:
                if ref_num not in refid2docid:
                    docid = "This is an irrelevant paper."
                else:
                    docid = refid2docid[ref_num]
                sen_1 = None
                sen_sentence = None
                sen_section = None
                if isinstance(docid,int):
                    tmp_1 = self.corpus_map[docid]['Title']
                    tmp_2 = self.corpus_map[docid]['Abstract']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: {tmp_2}"
                    sen_section = f"The paper titled '{tmp_1}' with the given abstract is relevant to the section: '{subtitle}'."
                    sen_sentence = f"The paper titled '{tmp_1}' with the given abstract could be cited in the sentence: '{sentence}'."
                else:
                    # The title
                    
                    # sen_1 = f"There is a paper. Title: '{docid}'. The title '{docid}' describes the content of the paper."
                    # sen_section = f"The paper titled '{docid}' is relevant to the section: '{subtitle}'."
                    # sen_sentence = f"The paper titled '{docid}' could be cited in the sentence: '{sentence}'."
                    sen_1 = "[NOTEXIST]"
                    sen_section = "[NOTEXIST]"
                    sen_sentence = "[NOTEXIST]"
                nli_pairs_sentence.append((sen_1,sen_sentence))
                nli_pairs_subtitle.append((sen_1,sen_section))
            section_relevance = None
            sentence_relevance = None
            if len(extracted_cites) > 0:    
                section_relevance = informationFuncs.eval_relevance_section(nli_pairs_subtitle,self.nli_model)
                sentence_relevance = informationFuncs.eval_relevance_sentence(nli_pairs_sentence,self.nli_model)
            else:
                sentence_relevance = 0
                section_relevance = 0
            eval_result["Information_Collection"]["Relevance"]["Section_Level"] = section_relevance
            eval_result["Information_Collection"]["Relevance"]["Sentence_Level"] = sentence_relevance
        elif "Relevance-Section" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
            nli_pairs_subtitle = []
            for ref_num,subtitle,sentence in extracted_cites:
                if ref_num not in refid2docid:
                    docid = "This is an irrelevant paper."
                else:
                    docid = refid2docid[ref_num]
                sen_1 = None
                if isinstance(docid,int):
                    tmp_1 = self.corpus_map[docid]['Title']
                    tmp_2 = self.corpus_map[docid]['Abstract']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: {tmp_2}"
                    sen_section = f"The paper titled '{tmp_1}' with the given abstract is relevant to the section: '{subtitle}'."
                else:
                    # The title
                    
                    # sen_1 = f"There is a paper. Title: '{docid}'. The title '{docid}' describes the content of the paper."
                    # sen_section = f"The paper titled '{docid}' is relevant to the section: '{subtitle}'." 
                    sen_1 = "[NOTEXIST]"
                    sen_section = "[NOTEXIST]"

                nli_pairs_subtitle.append((sen_1,sen_section))
            section_relevance = None
            if len(extracted_cites) > 0:    
                section_relevance = informationFuncs.eval_relevance_section(nli_pairs_subtitle,self.nli_model)
            else:
                section_relevance = 0
            
            eval_result["Information_Collection"]["Relevance"]["Section_Level"] = section_relevance
        elif "Relevance-Sentence" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
            nli_pairs_sentence = []
            for ref_num,subtitle,sentence in extracted_cites:
                if ref_num not in refid2docid:
                    docid = "This is an irrelevant paper."
                else:
                    docid = refid2docid[ref_num]
                sen_1 = None
                if isinstance(docid,int):
                    tmp_1 = self.corpus_map[docid]['Title']
                    tmp_2 = self.corpus_map[docid]['Abstract']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: {tmp_2}"
                    sen_sentence = f"The paper titled '{tmp_1}' with the given abstract could be cited in the sentence: '{sentence}'."
                else:
                    # The title
                    
                    # sen_1 = f"There is a paper. Title: '{docid}'. The title '{docid}' describes the content of the paper."
                    # sen_sentence = f"The paper titled '{docid}' could be cited in the sentence: '{sentence}'."
                    sen_1 = "[NOTEXIST]"
                    sen_sentence = "[NOTEXIST]"
                nli_pairs_sentence.append((sen_1,sen_sentence))
            
            sentence_relevance = None
            if len(extracted_cites) > 0:    
                sentence_relevance = informationFuncs.eval_relevance_sentence(nli_pairs_sentence,self.nli_model)
            else:
                sentence_relevance = 0
            sentence_relevance = informationFuncs.eval_relevance_sentence(nli_pairs_sentence,self.nli_model)
            eval_result["Information_Collection"]["Relevance"]["Sentence_Level"] = sentence_relevance
            
        if "Logic" in eval_list or "ALL" in eval_list:
            # if self.judge_model == None and self.using_openai == False:
            #     self.judge_model = AutoModelForCausalLM.from_pretrained(
            #         self.judge_model_path,
            #         torch_dtype= torch.float16,
            #         device_map="auto"
            #     )
            if self.using_openai == True:
                logic = informationFuncs.eval_logic_client(psg_node,self.client)
            else:
                pass 
                #logic = informationFuncs.eval_logic(psg_node,self.judge_model,self.judge_model_tokenizer)
            eval_result["Survey_Content"]["Logic"] = logic   
        
        return eval_result
            
        
    def eval_all(self,passage_dir,eval_list,save_path = None):
        if save_path != None :
            with open (save_path,"w",encoding='utf-8') as f:
                f.write('')
                
        survey_ids = [
            d for d in os.listdir(passage_dir)
            if os.path.isdir(os.path.join(passage_dir, d))
        ]

        length = len(survey_ids)
        
        eval_result = {
            "Information_Collection": {
                "Comprehensiveness": {
                    "Coverage": None,
                },
                "Relevance": {
                    "Paper_Level": None,
                    "Section_Level": None,
                    "Sentence_Level": None,
                }
            },
            "Survey_Structure": {
                "Structure_Quality(LLM_as_judge)": None,
                "SH-Recall": None
            },
            "Survey_Content": {
                "Relevance": {
                        "ROUGE-1": None,
                        "ROUGE-2": None,
                        "ROUGE-L": None,
                        "BLEU": None,
                    },
                "Logic": None
            }
        }

        
        for survey_id in tqdm(survey_ids):
            survey_dir = os.path.join(passage_dir,survey_id)
            
            psg_files = os.listdir(survey_dir)
            tmp_res = None
            
            if len(psg_files) == 1:
                psg_path = os.path.join(survey_dir,psg_files[0])
                tmp_res = self.single_eval(int(survey_id),psg_path,eval_list)
            else:
                for psg_file in psg_files:
                    if psg_file.endswith('.md'):
                        psg_path = os.path.join(survey_dir,psg_file)
                        tmp_res = self.single_eval(int(survey_id),psg_path,eval_list)
                        break
            # print(tmp_res)

            
            for k1,v1 in tmp_res.items():
                for k2,v2 in v1.items():
                    if isinstance(v2,dict):
                        for k3,v3 in v2.items():    
                            if v3 != None:
                                if eval_result[k1][k2][k3] == None:
                                    eval_result[k1][k2][k3] = v3/length
                                else:
                                    eval_result[k1][k2][k3] += v3/length
                    else:
                        if v2 != None:
                            if eval_result[k1][k2] == None:
                                eval_result[k1][k2] = v2/length
                            else:
                                eval_result[k1][k2] += v2/length
                                
            if save_path != None:
                tmp_res['survey_id'] = survey_id
                with open (save_path,"a",encoding='utf-8') as f:
                    json.dump(tmp_res,f,ensure_ascii=False,indent=4)
                    f.write('\n')
                print(f"Result of {survey_id}")
                print(tmp_res)
            else:
                print(f"Result of {survey_id}")
                print(tmp_res)    
        
        if save_path != None:
            eval_result['survey_id'] = "Average"
            with open (save_path,"a",encoding='utf-8') as f:
                f.write('\n')
                json.dump(eval_result,f,ensure_ascii=False,indent=4)
                print("Total Result:")
                print(eval_result)
        else:
            print("Total Result:")
            print(eval_result)
            
        return eval_result