from pathlib import Path
import datasets
from tree_sitter import Language, Parser
import tree_sitter_python
from pprint import pprint
import io, re
import tokenize
import random

PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

def prepare() -> datasets.Dataset:
    dataset = load_codesearch()
    dataset = dataset.map(add_extracted_fields)
    #dataset = load_dataset('D:/python-dataset/python_mod.ds')
    #print('loaded')
    #pprint(dataset)
    #demo_dataset(dataset)
    return dataset

# получить имя, тело и тело без комментариев в виде словаря для добавления к записи датасета
def add_extracted_fields(example):
    wfs = example['whole_func_string']
    func_name, body_clean, body_full = parse_function(wfs)
    return {
        **example,
        'extracted_func_name': func_name,
        'extracted_body_clean': body_clean,
        'extracted_body_full': body_full
    }

def demo_dataset(dataset : datasets.Dataset):
    print('Random examples of extraction function details\n')
    random_examples = random.sample(range(len(dataset)), 10)
    for idx in random_examples:
        example = dataset[idx]
        original_func_name = example['func_name'].split('.')[-1]
        original_documentation = example['whole_func_string']
        extracted_func_name = example['extracted_func_name']
        extracted_body_full = example['extracted_body_full']
        print(f"Original Function Name: {original_func_name}")
        print(f"Extracted Function Name: {extracted_func_name}")
        print(f"\nOriginal Body:\n{original_documentation}\n")
        print(f"\nExtracted Body:\n{extracted_body_full}\n")

        name_match = original_func_name == extracted_func_name
        doc_match = original_documentation.strip() == extracted_body_full.strip()
        #print(f"Function Name Match: {name_match}")
        #print(f"Body Match: {doc_match}")
        print("-" * 80)

# извлечь из whole_func_string название, тело, тело без документации и комментариев
def parse_function(whole_func_string : str):
    func_name = None
    body_full = None
    body_clean = None

    tree = parser.parse(bytes(whole_func_string, "utf8"))
    root_node = tree.root_node

    for node in root_node.children:
        if node.type == 'function_definition':
            func_name_node = node.child_by_field_name('name')
            if func_name_node:
                func_name = func_name_node.text.decode('utf8')
            # извлечение тела функции
            body_node = node.child_by_field_name('body')
            if body_node:
                body_full = body_node.text.decode('utf8')
                body_clean = clear_func_body(body_full) # удаление комментариев и документации
    return func_name, body_clean, body_full


# очистить тело функции от комментариев и документации
def clear_func_body(body : str):
    # работа со строкой как с файловым объектом
    stream = io.StringIO(body)
    body_clear = "" 
    previous_token_type = tokenize.INDENT  # тип предыдущего токена
    i = -1  # номер строки
    j = 0  # текущая позиция в строке
    # генерируем токены из входного кода
    for token in tokenize.generate_tokens(stream.readline):
        current_token_type = token[0]  # тип текущего токена
        token_value = token[1]  # строковое значение токена
        start_position, end_position = token[2], token[3]  # позиции начала и конца токена
        # если текущая строка больше предыдущей, сбрасываем позицию столбца
        if start_position[0] > i:
            j = 0
        # если текущий столбец больше предыдущего, добавляем пробелы для выравнивания
        if start_position[1] > j:
            body_clear += " " * (start_position[1] - j)
        # если токен — это комментарий, пропускаем его
        if current_token_type == tokenize.COMMENT:
            continue  
        # если токен — это строка
        elif current_token_type == tokenize.STRING:
            # проверяем, не находится ли он на новом уровне отступа или не является ли он новой строкой
            if previous_token_type != tokenize.INDENT and previous_token_type != tokenize.NEWLINE:
                if start_position[1] > 0:  body_clear += token_value
        else:
            body_clear += token_value  # добавляем все остальные токены
        previous_token_type = current_token_type
        j = end_position[1] 
        i = end_position[0]
    return '\n'.join(line for line in body_clear.splitlines() if line.strip())

def load_codesearch():
    dataset = datasets.load_dataset('code-search-net/code_search_net', name='python', trust_remote_code=True, split='test[:1000]')
    return dataset

def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))

def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))