import requests
import time
import csv
import pandas
import re


def get_chatgpt_request(content):

    # Set the API key
    api_key = ""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.7
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()
def Run_ChatGPT_by_prompt(prompt):

    print("input: ", prompt)

    max_loop = 30
    INDEX = 0
    while(True):
        INDEX += 1
        if INDEX > max_loop:
            print("connect error")
            assert False

        print("start the ", INDEX, "request")
        try:
            answer = get_chatgpt_request(prompt)
            answer = answer['choices'][0]['message']['content']
            print("output: ", answer)
            break
        except:
            print("request error and sleep 3s")
            time.sleep(3)
    return answer


def get_dataset_reason():
    f = open("VUAPOS_train.csv", "r")
    fr = csv.reader(f)

    fw = csv.writer(open("VUAPOStrain_reason.csv", "a"))

    for row in fr:
        word = row[4]
        sent = row[1]
        label = int(row[0])

        prompt_1 = f'"{word}" is used metaphorically in "{sent}", give reasons why (15 words or less)'
        prompt_0 = f'"{word}" is used non-metaphorically in "{sent}", give reasons why (15 words or less)'

        if label == 0:
            answer = Run_ChatGPT_by_prompt(prompt_0)
        elif label == 1:
            answer = Run_ChatGPT_by_prompt(prompt_1)
        else:
            assert False

        fw.writerow(row + [answer])

def get_final_dataset_reason():
    f = open("MOH-X_reason.csv", "r")
    fr = csv.reader(f)

    fw = csv.writer(open("MOH-X_reason2.csv", "w"))

    for row in fr:
        word = row[1]
        sent = row[3]
        label = row[4]
        reason = row[5]

        if label == "1":
            usage = "metaphor"
        elif label == "0":
            usage = "literal"
        else:
            assert False

        final_answer_usage = "Usage:" + usage

        final_answer_both = "Usage:" + usage + "\nReason:" + reason

        fw.writerow(row[:-1] + [final_answer_usage, final_answer_both])


def get_data_split():

    import pandas as pd
    import random

    random.seed(45)

    data = pd.read_csv('TroFi_reason2.csv')

    unique_verbs = data['word'].unique()
    unique_label = data['label'].unique()

    train_data, test_data = pd.DataFrame(), pd.DataFrame()

    for verb in unique_verbs:
            for label in unique_label:

                verb_samples = data[(data['word'] == verb) & (data['label'] == label)]

                total_samples = len(verb_samples)
                num_train = int(total_samples * 0.5)

                if total_samples == 1:

                    num = random.random()
                    if num < 0.65:
                        train_data = pd.concat([train_data, verb_samples])
                    else:
                        test_data = pd.concat([test_data, verb_samples])

                else:
                    verb_samples = verb_samples.sample(frac=1)

                    train_samples = verb_samples[:num_train]
                    train_data = pd.concat([train_data, train_samples])

                    test_samples = verb_samples[num_train:]
                    test_data = pd.concat([test_data, test_samples])


    print(train_data.shape[0], test_data.shape[0], train_data.shape[0] + test_data.shape[0])

    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)

def cal_metrics(labels, preds):
    true_positives = sum(1 for t, p in zip(labels, preds) if t == p and t == 1)
    false_positives = sum(1 for t, p in zip(labels, preds) if t != p and t == 0)
    true_negatives = sum(1 for t, p in zip(labels, preds) if t == p and t == 0)
    false_negatives = sum(1 for t, p in zip(labels, preds) if t != p and t == 1)

    accuracy = (true_positives + true_negatives) / len(labels)
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1_score)


def cal_gpt2_output_metrics():
    f = open("gpt2_results/trofi_gpt2_120.txt", "r")
    content = f.read()

    sample_list = content.split("------------------------------------------------------")

    total_num = 0

    preds = []
    labels = []

    for sample in sample_list:
        if sample.strip():

            prompt = sample.split("pred:")[0]
            pred = sample.split("pred:")[1].split("label:")[0]
            label = sample.split("pred:")[1].split("label:")[1]

            new_pred = []
            pred_list = pred.split("\n")
            for pl in pred_list:
                if pl:
                    new_pred.append(pl)

            if "Usage:" in new_pred[0] and "Reason:" in new_pred[1]:
                temp_usage = re.sub("Usage:", "", new_pred[0])
                temp_reason = re.sub("Reason:", "", new_pred[1])
            elif "Usage:" in new_pred[1] and "Reason:" in new_pred[2]:
                temp_usage = re.sub("Usage:", "", new_pred[1])
                temp_reason = re.sub("Reason:", "", new_pred[2])
            else:
                print(new_pred)
                assert False

            if temp_usage.lower() == "metaphor":
                final_pred = 1
            elif temp_usage.lower() == "literal":
                final_pred = 0
            else:
                assert False

            if len(temp_reason.split(".Is")) > 1:
                final_reason = temp_reason.split(".Is")[0]
            elif len(temp_reason.split(".Is")) == 1:
                final_reason = temp_reason.split(".Is")[0]
            else:
                assert False

            preds.append(final_pred)

            new_label = []
            label_list = label.split("\n")
            for ll in label_list:
                if ll:
                    new_label.append(ll)

            if "Usage:" in new_label[0] and "Reason:" in new_label[1]:
                temp_usage = re.sub("Usage:", "", new_label[0])
                temp_reason = re.sub("Reason:", "", new_label[1])
            else:
                print(new_label)
                assert False

            if temp_usage.lower() == "metaphor":
                final_label = 1
            elif temp_usage.lower() == "literal":
                final_label = 0
            else:
                assert False

            labels.append(final_label)

            print(final_pred, final_label)
            total_num += 1

    print(total_num)
    cal_metrics(labels, preds)



def Run_LLM_reasoning():
    f = open("VUAPOS_reason/test.csv", "r")
    fr = csv.reader(f)

    fw = csv.writer(open("VUAPOS_test_chatgpt.csv", "a"))

    for i, row in enumerate(fr):

        if i == 0:
            fw.writerow(row + ["pred"])
            continue


        word = row[1]
        sent = row[3]

        prompt = f'Determine whether “{word}” is used metaphorically or literally in “{sent}” and give reasons why.\nUsage:(metaphor or literal)\nReason:(15 words or less)'
        answer = Run_ChatGPT_by_prompt(prompt)

        fw.writerow(row + [answer])

def Run_LLM_reason_usage_only():
    f = open("MOX_reason/test.csv", "r")
    fr = csv.reader(f)

    fw = csv.writer(open("MOX_test_chatgpt_label.csv", "w"))

    for i, row in enumerate(fr):

        if i == 0:
            fw.writerow(row + ["pred"])
            continue

        word = row[1]
        sent = row[3]

        prompt = f'Determine whether “{word}” is used metaphorically or literally in “{sent}”\nUsage:(metaphor or literal)'
        answer = Run_ChatGPT_by_prompt(prompt)

        fw.writerow(row + [answer])


def cal_llama3_pred_metrics():

    f = open("VUAPOS_test_chatgpt.csv", "r")
    fr = csv.reader(f)

    labels = []
    preds = []

    for i, row in enumerate(fr):
        if i == 0:
            continue

        pred = row[7]
        label = int(row[4])

        new_pred = []
        pred_list = pred.split("\n")
        for pl in pred_list:
            if pl:
                new_pred.append(pl)

        if len(new_pred) < 2:
            new_pred = pred.split(".")
            if len(new_pred) < 2:
                print("pred len error:", new_pred)
                continue

        if "Usage:" in new_pred[0] and "Reason:" in new_pred[1]:
            temp_usage = re.sub("Usage:", "", new_pred[0]).strip()
            temp_reason = re.sub("Reason:", "", new_pred[1]).strip()
        elif "Usage:" in new_pred[0] and "Reason:" not in new_pred[1]:
            temp_usage = re.sub("Usage:", "", new_pred[0]).strip()
            temp_reason = new_pred[1].strip()
        elif "Usage:" not in new_pred[0] and "Reason:" in new_pred[1]:
            temp_usage = new_pred[0].strip()
            temp_reason = re.sub("Reason:", "", new_pred[1]).strip()
        else:
            temp_usage = new_pred[0].strip()
            temp_reason = new_pred[1].strip()

        temp_usage = temp_usage.rstrip(string.punctuation)

        if temp_usage.lower() == "metaphor" or temp_usage.lower() == "metaphorical" or temp_usage.lower() == "metaphorically":
            final_pred = 1
        elif temp_usage.lower() == "literal" or temp_usage.lower() == "literally":
            final_pred = 0
        elif "metaphorically" in temp_usage.lower() or "metaphorical" in temp_usage.lower():
            final_pred = 1
        else:
            if "literal" in temp_usage.lower() and "metaphor" in temp_usage.lower():
                continue
            if temp_usage.lower() == "not present":
                continue
            print(pred)
            continue


        labels.append(label)
        preds.append(final_pred)

    cal_metrics(labels, preds)


def cal_llama3_label_metrics():

    f = open("MOX_test_chatgpt_label.csv", "r")
    fr = csv.reader(f)

    labels = []
    preds = []

    for i, row in enumerate(fr):
        if i == 0:
            continue

        pred = row[7]
        label = int(row[4])

        temp_usage = pred.strip()

        if temp_usage.lower() == "metaphor" or temp_usage.lower() == "metaphorical" or temp_usage.lower() == "metaphorically":
            final_pred = 1
        elif temp_usage.lower() == "literal" or temp_usage.lower() == "literally":
            final_pred = 0
        elif "metaphorically" in temp_usage.lower() or "metaphorical" in temp_usage.lower():
            final_pred = 1
        elif "literal" in temp_usage.lower():
            final_pred = 0
        else:
            print(temp_usage)
            assert False


        labels.append(label)
        preds.append(final_pred)

    cal_metrics(labels, preds)


def Run_similiary_pred():

    data_name = "MOX_reason"
    model_name = "chatgpt"

    f = open(data_name + "/total_" + model_name + ".csv", "r")
    fr = csv.reader(f)

    fw = csv.writer(open(data_name + "/similarity_" + model_name + ".csv", "w"))

    for i, row in enumerate(fr):

        if i == 0:
            fw.writerow(row + ["similiary"])
            continue

        word = row[1]
        sent = row[3]
        answer = row[5]
        pred = row[6]

        # word = row[0]
        # sent = row[1]
        # answer = row[3]
        # pred = row[4]

        prompt = f'Discuss the use of "{word}" in "{sent}".\nAnswer:{answer}\nPrediction:{pred}\nRate the prediction based on the answer (1 to 5).\nOutput:'
        answer = Run_ChatGPT_by_prompt(prompt)

        fw.writerow(row + [answer])


def Run_similiary_pred_wo_usage():

    data_name = "TroFi_reason"
    model_name = "llama3-70b"

    f = open(data_name + "/total_" + model_name + ".csv", "r")
    fr = csv.reader(f)

    fw = csv.writer(open(data_name + "/similarity_" + model_name + "_wo_usage.csv", "w"))

    for i, row in enumerate(fr):

        if i == 0:
            fw.writerow(row + ["similiary"])
            continue

        word = row[1]
        sent = row[3]
        answer = row[5]
        pred = row[6]

        # word = row[0]
        # sent = row[1]
        # answer = row[3]
        # pred = row[4]

        try:
            answer = answer.split(". ")[1]
            pred = pred.split(". ")[1]
        except:
            pass

        prompt = f'Discuss the use of "{word}" in "{sent}".\nAnswer:{answer}\nPrediction:{pred}\nRate the prediction based on the answer (1 to 5).\nOutput:'
        answer = Run_ChatGPT_by_prompt(prompt)

        fw.writerow(row + [answer])

def convert_gpt2_text_to_csv():

    data_name = "TroFi_reason"

    f = open(data_name + "/trofi_gpt2l.txt", "r")
    content = f.read()

    fw = csv.writer(open(data_name + "/test_gpt2l.csv", "w"))


    sample_list = content.split("------------------------------------------------------")
    total_num = 0

    for sample in sample_list:
        if sample.strip():
            # print(sample)

            prompt = sample.split("pred:")[0]
            pred = sample.split("pred:")[1].split("label:")[0]
            label = sample.split("pred:")[1].split("label:")[1]

            new_pred = []
            pred_list = pred.split("\n")
            for pl in pred_list:
                if pl:
                    new_pred.append(pl)

            if "Usage:" in new_pred[0] and "Reason:" in new_pred[1]:
                temp_usage = re.sub("Usage:", "", new_pred[0]).strip()
                temp_reason = re.sub("Reason:", "", new_pred[1]).strip()
            elif "Usage:" in new_pred[1] and "Reason:" in new_pred[2]:
                temp_usage = re.sub("Usage:", "", new_pred[1]).strip()
                temp_reason = re.sub("Reason:", "", new_pred[2]).strip()
            else:
                print(new_pred)
                assert False

            if len(temp_reason.split(".Is")) > 1:
                temp_reason = temp_reason.split(".Is")[0].strip()
            elif len(temp_reason.split(".Is")) == 1:
                temp_reason = temp_reason.split(".Is")[0].strip()
            else:
                assert False

            temp_usage = temp_usage.rstrip(string.punctuation).lower()
            temp_output = temp_usage + ". " + temp_reason

            # get word and sentence in prompt
            pattern = r'“([^”]*)”'
            matches = re.findall(pattern, prompt)
            assert len(matches) == 2
            temp_word = matches[0]
            temp_sent = matches[1]

            # cal final label
            new_label = []
            label_list = label.split("\n")
            for ll in label_list:
                if ll:
                    new_label.append(ll)

            if "Usage:" in new_label[0] and "Reason:" in new_label[1]:
                temp_usage = re.sub("Usage:", "", new_label[0])
                temp_reason = re.sub("Reason:", "", new_label[1])
            else:
                assert False

            if temp_usage.lower() == "metaphor":
                temp_label = 1
            elif temp_usage.lower() == "literal":
                temp_label = 0
            else:
                assert False

            temp_usage = temp_usage.strip().lower()
            temp_reason = temp_usage + ". " + temp_reason

            fw.writerow([temp_word, temp_sent, temp_label, temp_reason, temp_output])

def cal_LLMs_average_evaluate_score():
    model_name = "chatgpt"
    data_name = "MOX_reason"

    fr = csv.reader(open(data_name + "/similarity_" + model_name + ".csv", "r"))

    total_scores = []
    metaphor_scores = []
    literal_scores = []

    LABEL_INDEX = 4
    SIMILARITY_INDEX = 7

    for i, row in enumerate(fr):
        if i == 0:
            continue

        try:
            temp_label = int(row[LABEL_INDEX])
        except:
            continue

        try:
            temp_sim = int(row[SIMILARITY_INDEX])
        except:
            try:
                temp_sim = int(row[SIMILARITY_INDEX][0])
            except:
                continue
            print(temp_sim, "<-", row[SIMILARITY_INDEX])

        total_scores.append(temp_sim)
        if temp_label == 1:
            metaphor_scores.append(temp_sim)
        elif temp_label == 0:
            literal_scores.append(temp_sim)
        else:
            assert False


    total_ave = sum(total_scores) / len(total_scores)
    metaphor_ave = sum(metaphor_scores) / len(metaphor_scores)
    literal_ave = sum(literal_scores) / len(literal_scores)

    metaphor_rate = len(metaphor_scores) / (len(metaphor_scores) + len(literal_scores))
    literal_rate = len(literal_scores) / (len(metaphor_scores) + len(literal_scores))

    print("total:", total_ave, len(total_scores))
    print("metaphor:", metaphor_ave, len(metaphor_scores))
    print("literal:", literal_ave, len(literal_scores))

    print("total_weight:", metaphor_rate * metaphor_ave + literal_rate * literal_ave, metaphor_rate, literal_rate)


def cal_entail_average_evaluate_score():
    # model_name_list = ["gpt2l", "llama3-70b", "chatgpt"]
    model_name_list = ["chatgpt"]
    data_name_list = ["vua", "trofi", "mox"]
    entail_name_list = ["STS-B", "SICK"]


    for model_name in model_name_list:
        for data_name in data_name_list:
            for entail_name in entail_name_list:

                fr1 = csv.reader(open("entail_data/pred_datas/" + data_name + "_" + model_name + "_" + entail_name + ".csv",  "r"))
                fr2 = csv.reader(open("entail_data/ori_datas/" + data_name + "/" + model_name + ".csv", "r"))

                ori_datas = []
                for i, row in enumerate(fr2):
                    if i == 0:
                        continue
                    ori_datas.append(row)

                num = 0
                total_scores = []
                metaphor_scores = []
                literal_scores = []

                for i, row in enumerate(fr1):

                    for od in ori_datas:

                        if not od[0] and not od[1]:
                            continue

                        if row[:2] == od[:2]:

                            try:
                                temp_label = int(od[2])
                            except:
                                break

                            temp_sim = int(row[2]) + 1

                            total_scores.append(temp_sim)
                            if temp_label == 1:
                                metaphor_scores.append(temp_sim)
                            elif temp_label == 0:
                                literal_scores.append(temp_sim)
                            else:
                                print(temp_label)
                                assert False

                            num += 1
                            break

                print(data_name + "-" + model_name + "-" + entail_name, num)
                total_ave = sum(total_scores) / len(total_scores)
                metaphor_ave = sum(metaphor_scores) / len(metaphor_scores)
                literal_ave = sum(literal_scores) / len(literal_scores)

                metaphor_rate = len(metaphor_scores) / (len(metaphor_scores) + len(literal_scores))
                literal_rate = len(literal_scores) / (len(metaphor_scores) + len(literal_scores))

                print("total:", total_ave, len(total_scores))
                print("metaphor:", metaphor_ave, len(metaphor_scores))
                print("literal:", literal_ave, len(literal_scores))
                print("total_weight:", metaphor_rate * metaphor_ave + literal_rate * literal_ave, metaphor_rate, literal_rate)
def cal_entail_average_evaluate_score_wo_usage():

    fr1 = csv.reader(open("datas/entail_datas/trofi_llama3-70b_SICK_wo_usage.csv",  "r"))

    num = 0
    total_scores = []
    metaphor_scores = []
    literal_scores = []

    for i, row in enumerate(fr1):

        temp_sim = int(row[2]) + 1
        temp_label = int(row[3])

        total_scores.append(temp_sim)
        if temp_label == 1:
            metaphor_scores.append(temp_sim)
        elif temp_label == 0:
            literal_scores.append(temp_sim)
        else:
            print(temp_label)
            assert False


    total_ave = sum(total_scores) / len(total_scores)
    metaphor_ave = sum(metaphor_scores) / len(metaphor_scores)
    literal_ave = sum(literal_scores) / len(literal_scores)

    metaphor_rate = len(metaphor_scores) / (len(metaphor_scores) + len(literal_scores))
    literal_rate = len(literal_scores) / (len(metaphor_scores) + len(literal_scores))

    print("total:", total_ave, len(total_scores))
    print("metaphor:", metaphor_ave, len(metaphor_scores))
    print("literal:", literal_ave, len(literal_scores))
    print("total_weight:", metaphor_rate * metaphor_ave + literal_rate * literal_ave, metaphor_rate, literal_rate)


def convert_MRPC_txt_to_csv():
    f = open("MRPC/msr_paraphrase_train.txt", "r")
    row_list = f.read().split("\n")

    fw = csv.writer(open("MRPC/train.csv", "w"))

    for i, row in enumerate(row_list):
        if i == 0:
            fw.writerow(["sent1", "sent2", "label"])
            continue
        if not row:
            continue

        row_split = row.split("	")

        assert len(row_split) == 5

        label = int(row_split[0])
        sent1 = row_split[3]
        sent2 = row_split[4]

        print(sent1, sent2, label)
        fw.writerow([sent1, sent2, label])


def get_similarity_entail_data():
    data_name = "VUAPOS_reason"
    model_name = "chatgpt"

    fr = csv.reader(open(data_name + "/similarity_" + model_name + ".csv", "r"))
    fw = csv.writer(open(data_name + "/" + model_name + ".csv", "w"))

    for i, row in enumerate(fr):
        if i == 0:
            fw.writerow(["sent1", "sent2", "label"])
            continue

        # sent1 = row[3]
        # sent2 = row[4]
        # label = row[2]

        sent1 = row[5]
        sent2 = row[6]
        label = row[4]

        print(sent1, sent2, label)
        fw.writerow([sent1, sent2, label])



def cal_BLEU(reference, candidate):
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate import meteor_score
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', "rouge2", "rougeL"], use_stemmer=True)

    # bleu1 = sentence_bleu([reference.split()], candidate.split(), weights=(1, 0, 0, 0))
    # acc_bleu2 = sentence_bleu([reference.split()], candidate.split(), weights=(0.5, 0.5, 0, 0))
    # acc_bleu3 = sentence_bleu([reference.split()], candidate.split(), weights=(0.33, 0.33, 0.33, 0))
    # acc_bleu4 = sentence_bleu([reference.split()], candidate.split(), weights=(0.25, 0.25, 0.25, 0.25))

    bleu1 = sentence_bleu([reference.split()], candidate.split(), weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu([reference.split()], candidate.split(), weights=(0, 1, 0, 0))
    bleu3 = sentence_bleu([reference.split()], candidate.split(), weights=(0, 0, 1, 0))
    bleu4 = sentence_bleu([reference.split()], candidate.split(), weights=(0, 0, 0, 1))
    meteor = meteor_score.meteor_score([reference.split()], candidate.split())
    rouge_score = scorer.score(reference, candidate)
    rouge1 = rouge_score["rouge1"].recall
    rouge2 = rouge_score["rouge2"].recall
    rougeL = rouge_score["rougeL"].recall

    return [bleu1, bleu2, bleu3, bleu4, rouge1, rouge2, rougeL, meteor]


def Run_BLEU_scores():
    model_name_list = ["gpt2l", "llama3-70b", "chatgpt"]
    data_name_list = ["vua", "trofi", "mox"]

    for model_name in model_name_list:
        for data_name in data_name_list:

            fr = csv.reader(open("datas/ori_datas/" + data_name + "/" + model_name + ".csv", "r"))
            fw = csv.writer(open("datas/bleu_datas/" + data_name + "/" + model_name + ".csv", "w"))

            for i, row in enumerate(fr):
                if i == 0:
                    fw.writerow(row + ["matrics"])
                    continue

                sent1 = row[0]
                sent2 = row[1]
                label = row[2]

                metrics_list = cal_BLEU(sent1, sent2)
                print(sent1, " <-> ", sent2)
                print(metrics_list)

                fw.writerow([sent1, sent2, label, metrics_list])


def Run_BLEU_scores_wo_usage():
    fr = csv.reader(open("datas/ori_datas/trofi/gpt2l.csv", "r"))
    fw = csv.writer(open("datas/bleu_datas/trofi/gpt2l_.csv", "w"))

    for i, row in enumerate(fr):
        if i == 0:
            fw.writerow(row + ["matrics"])
            continue

        sent1 = row[0]
        sent2 = row[1]

        sent1 = sent1.split(". ")[1]
        sent2 = sent2.split(". ")[1]

        label = row[2]

        metrics_list = cal_BLEU(sent1, sent2)
        print(sent1, " <-> ", sent2)
        print(metrics_list)

        fw.writerow([sent1, sent2, label, metrics_list])


def get_BLEU_ave_scores():
    model_name_list = ["gpt2l", "llama3-70b", "chatgpt"]
    data_name_list = ["vua", "trofi", "mox"]

    gpt2l_scores = [0] * 3 * 8
    chatgpt_scores = [] * 3 * 8
    llama3_scores = [] * 3 * 8

    for model_name in model_name_list:
        for data_name in data_name_list:

            fr = csv.reader(open("datas/bleu_datas/" + data_name + "/" + model_name + ".csv", "r"))

            score_labels = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
            score_dir = {}

            for i, row in enumerate(fr):
                if i == 0:
                    continue

                matrics_list = eval(row[3])

                for ii, metrics in enumerate(matrics_list):
                    score_label = score_labels[ii]
                    if score_label not in score_dir.keys():
                        score_dir[score_label] = [float(metrics)]
                    else:
                        score_dir[score_label].append(float(metrics))


            print(data_name, ",", model_name)

            temp_score = []
            for key in score_dir.keys():
                print(key)
                score = score_dir[key]
                print(sum(score) / len(score))
                temp_score.append(round(sum(score) / len(score), 2))

            if model_name == "gpt2l":
                if data_name == "vua":
                    gpt2l_scores[0:8] = temp_score
                elif data_name == "trofi":
                    gpt2l_scores[8:16] = temp_score
                else:
                    gpt2l_scores[16:24] = temp_score
            elif model_name == "chatgpt":
                if data_name == "vua":
                    chatgpt_scores[0:8] = temp_score
                elif data_name == "trofi":
                    chatgpt_scores[8:16] = temp_score
                else:
                    chatgpt_scores[16:24] = temp_score
            else:
                if data_name == "vua":
                    llama3_scores[0:8] = temp_score
                elif data_name == "trofi":
                    llama3_scores[8:16] = temp_score
                else:
                    llama3_scores[16:24] = temp_score

    matrix = [[i * 24 + j for j in range(24)] for i in range(4)]

    matrix[0][:] = score_labels * 3
    matrix[1][:] = gpt2l_scores
    matrix[2][:] = chatgpt_scores
    matrix[3][:] = llama3_scores

    fw = csv.writer(open("bleu.csv", "w"))
    for row in matrix:
        print(row)
        fw.writerow(row)


def cal_llama_and_chatgpt_VUAverb_similiary():
    fr = csv.reader(open("VUAPOS_reason/similarity_llama3-70b.csv", "r"))

    m_num = 0
    l_num = 0

    m_sim = 0
    l_sim = 0

    for i, row in enumerate(fr):
        if i == 0:
            continue

        pos = row[2]
        try:
            label = int(row[4])
        except:
            continue
        try:
            sim = int(row[7])
        except:
            try:
                sim = int(row[7][0])
            except:
                continue

        if pos != "VERB":
            continue

        if label == 0:
            l_num += 1
            l_sim += sim
        elif label == 1:
            m_num += 1
            m_sim += sim

    l_rate = l_num / (l_num + m_num)
    m_rate = m_num / (l_num + m_num)

    print(l_num, m_num, l_num + m_num, l_rate, m_rate)

    l_score = l_sim / l_num
    m_score = m_sim / m_num

    print(round(m_score, 3))
    print(round(l_score, 3))
    print(round(l_score * l_rate + m_score * m_rate, 3))


def cal_gpt2l_VUAverb_similiary_by_llama():
    fr = csv.reader(open("VUAPOS_reason/similarity_gpt2l.csv", "r"))
    fr_ = csv.reader(open("VUAPOS_reason/similarity_chatgpt.csv", "r"))

    row_list = [row for row in fr_]

    m_num = 0
    l_num = 0

    m_sim = 0
    l_sim = 0

    for i, row in enumerate(fr):
        if i == 0:
            continue

        word = row[0]
        sent = row[1]

        try:
            label = int(row[2])
        except:
            continue
        try:
            sim = int(row[5])
        except:
            try:
                sim = int(row[5][0])
            except:
                continue

        pos = ""
        for rl in row_list:
            rl_word = rl[1]
            rl_pos = rl[2]
            rl_sent = rl[3]

            if rl_word.strip() == word.strip() and rl_sent.strip() == sent.strip():
                pos = rl_pos
                print(rl_pos)
                break

        if pos != "VERB" or not pos:
            continue

        if label == 0:
            l_num += 1
            l_sim += sim
        elif label == 1:
            m_num += 1
            m_sim += sim

    l_rate = l_num / (l_num + m_num)
    m_rate = m_num / (l_num + m_num)

    print(l_num, m_num, l_num + m_num, l_rate, m_rate)

    l_score = l_sim / l_num
    m_score = m_sim / m_num

    print(round(m_score, 3))
    print(round(l_score, 3))
    print(round(l_score * l_rate + m_score * m_rate, 3))

def cal_LLM_VUAverb_entail():
    fr_ = csv.reader(open("VUAPOS_reason/similarity_llama3-70b.csv", "r"))
    row_list = [row for row in fr_]

    fr = csv.reader(open("datas/entail_datas/vua_gpt2l_STS-B.csv", "r"))


    m_num = 0
    l_num = 0

    m_sim = 0
    l_sim = 0
    for i, row in enumerate(fr):
        sent_ori = row[0]
        sent_l = row[1]
        score = int(row[2]) + 1

        label = -1
        pos = ""

        for rl in row_list:
            rl_pos = rl[2]
            rl_sent = rl[5]
            rl_label = rl[4]

            if rl_sent.strip() == sent_ori.strip():
                pos = rl_pos
                label = rl_label
                break

        try:
            label = int(label)
        except:
            continue

        if pos != "VERB" or not pos:
            continue

        print(score, label, pos, sent_ori, sent_l)

        if label == 0:
            l_num += 1
            l_sim += score
        elif label == 1:
            m_num += 1
            m_sim += score

    l_rate = l_num / (l_num + m_num)
    m_rate = m_num / (l_num + m_num)

    print(l_num, m_num, l_num + m_num, l_rate, m_rate)

    l_score = l_sim / l_num
    m_score = m_sim / m_num

    print("Met", round(m_score, 3))
    print("Lit", round(l_score, 3))
    print("Wtd", round(l_score * l_rate + m_score * m_rate, 3))

def cal_LLM_VUAverb_bleu():
    fr_ = csv.reader(open("VUAPOS_reason/similarity_llama3-70b.csv", "r"))
    row_list = [row for row in fr_]

    fr = csv.reader(open("datas/bleu_datas/vua/gpt2l.csv", "r"))

    SCORE_LIST = [0] * 8
    TOTAL_NUM = 0

    for i, row in enumerate(fr):
        if i == 0:
            continue
        sent_ori = row[0]
        sent_l = row[1]
        score_list = eval(row[3])

        pos = ""
        for rl in row_list:
            rl_pos = rl[2]
            rl_sent = rl[5]

            if rl_sent.strip() == sent_ori.strip():
                pos = rl_pos
                break

        if pos != "VERB" or not pos:
            continue

        SCORE_LIST = [x + y for x, y in zip(SCORE_LIST, score_list)]
        TOTAL_NUM += 1

        print(SCORE_LIST, TOTAL_NUM)

    ave_scores = [round(x / TOTAL_NUM, 3) for x in SCORE_LIST]

    print(ave_scores)













