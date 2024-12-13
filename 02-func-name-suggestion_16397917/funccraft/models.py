from collections.abc import Iterable
from functools import cache
from pprint import pprint

import datasets
import evaluate
from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration
import time
import random

@cache
def _init_metrics():
    return (evaluate.load('exact_match'), evaluate.load('rouge'))

def predict(dataset: datasets.Dataset, model_name: str,
            comments_include=False, time_it=False,  show_precidts=False, eval_worst=0) -> None:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to('cpu')

    # получем тела функций для разбора - либо чистые, либо с документацией
    if comments_include:
        function_bodies = [f"{example['extracted_body_full']}" for example in dataset]
    else:
        function_bodies = [f"{example['extracted_body_clean']}" for example in dataset]
    # реальные названия функций
    true_names = [example['extracted_func_name'] for example in dataset]

    if time_it:
        start_time = time.time()
        print()
    #function_bodies = function_bodies[:10] # сокращенный пример
    #true_names = true_names[:10]
    # генерация предсказаний
    predictions = []
    for body in function_bodies:
        input_text = body
        # для модели Salesforce/codet5p-220m добавим токен <extra_id_0> для предсказываемого имени
        if model_name == 'Salesforce/codet5p-220m':
            input_text = 'def <extra_id_0>(): ' + body
        inputs = tokenizer.encode(input_text, return_tensors="pt").to('cpu')
        outputs = model.generate(inputs, max_length=30)
        pred_names = tokenizer.decode(outputs[0], skip_special_tokens=True).split()
        if len(pred_names) > 0:
            pred_name = pred_names[0].split('(')[0]
        else:
            pred_name = ''
        predictions.append(pred_name)
        #print('\nPredicted name: ' + pred_name)
        #print('True name: ' + true_names[len(predictions)-1])
    if time_it: # если фиксируем время
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Predictions completed in {minutes} min {seconds} s")
    if show_precidts: # если нужно продемонстрировать, берем 10 примеров
        print("\nExamples of predictions:")
        for _ in range(10):
            idx = random.randint(0, len(predictions)-1)
            print(f"{true_names[idx]} predicted as {predictions[idx]}")
    # оценка результатов
    eval_results = run_evaluate(predictions=predictions, references=true_names)

    print()
    print('*' * 80)
    print(f'\nEvaluation results (including comments: {comments_include}):')
    pprint(eval_results)
    print('*' * 80)
    print()

    if (eval_worst == 0): # если без выборки худших, то заканчиваем
        return
    # получение худших случаевв
    rouge = evaluate.load("rouge")
    results = []
    for i in range(len(predictions)):
        pred, true = predictions[i], true_names[i] # отдельная пара
        rouge_score = rouge.compute(predictions=[pred], references=[true])['rouge1']
        results.append((pred, true, function_bodies[i], rouge_score))
    # сортировка по значению оценки и отсечение нужного количества худших
    results.sort(key=lambda x: x[3])
    worst_predictions = results[:eval_worst]
    print()
    print('*' * 80)
    print(f"\nThe worst predictions:")
    print()
    for pred, true, body, rouge_score in worst_predictions:
        print('\nPredicted name: ' + pred)
        print('True name: ' + true)
        print(f'ROUGE-1: {rouge_score}')
        print()
        print(body)
        print()
    print('*' * 80)
    print()

def run_evaluate(
    predictions: Iterable[str], references: Iterable[str]
) -> dict[str, float]:
    em, rouge = _init_metrics()
    em_score = em.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return {**rouge_scores, **em_score}
