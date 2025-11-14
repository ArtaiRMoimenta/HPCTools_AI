# Implementación de alexaapo sacada de https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset

# Cargar modulos necesarios
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import time
import transformers
transformers.AdamW = torch.optim.AdamW #En la ultima version de transformers se saco AdamW. La llamamos desde torch  
from lightning_fabric.fabric import Fabric

 # PAsar strategia por linea de comandos

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, default='ddp', help='Distributed strategy for Fabric')
args = parser.parse_args()



# Almacenar textos, consultas y respuestas de los archivos .json de entrenamiento (train) y validación (dev). 
# Guardamos esta información en listas. Copiado directamente de su implementacion

# Give the path for train data
path = Path('squad/train-v2.0.json')

# Open .json file
with open(path, 'rb') as f:
    squad_dict = json.load(f)

texts = []
queries = []
answers = []

# Search for each passage, its question and its answer
for group in squad_dict['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            for answer in qa['answers']:
                # Store every passage, query and its answer to the lists
                texts.append(context)
                queries.append(question)
                answers.append(answer)

train_texts, train_queries, train_answers = texts, queries, answers

# Give the path for validation data
path = Path('squad/dev-v2.0.json')

# Open .json file
with open(path, 'rb') as f:
    squad_dict = json.load(f)

texts = []
queries = []
answers = []

# Search for each passage, its question and its answer
for group in squad_dict['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            for answer in qa['answers']:
                # Store every passage, query and its answer to the lists
                texts.append(context)
                queries.append(question)
                answers.append(answer)

val_texts, val_queries, val_answers = texts, queries, answers

# Comentario a la implementacion usada:
# Because Bert model needs both start and end position characters of the answer, I have to find it and store it for later. 
# Sometimes, I notice that SQuAD anwers "eat" one or two characters from the real answer in the passage. 
#For example, (as a colleague said in Piazza) for the word "sixth" in passage, SQuAD give the answer of "six". 
#So in these cases I select to handle this problem by "cutting" the passage by 1 or 2 characters to be the same as the given answer. 
#This strategy is because BERT works with tokens of a specific format so I needed to process the squad dataset to keep up with the input that BERT is waiting for.

# Find end position character in train data
for answer, text in zip(train_answers, train_texts):
    real_answer = answer['text']
    start_idx = answer['answer_start']
    # Get the real end index
    end_idx = start_idx + len(real_answer)

    # Deal with the problem of 1 or 2 more characters 
    if text[start_idx:end_idx] == real_answer:
        answer['answer_end'] = end_idx
    # When the real answer is more by one character
    elif text[start_idx-1:end_idx-1] == real_answer:
        answer['answer_start'] = start_idx - 1
        answer['answer_end'] = end_idx - 1  
    # When the real answer is more by two characters  
    elif text[start_idx-2:end_idx-2] == real_answer:
        answer['answer_start'] = start_idx - 2
        answer['answer_end'] = end_idx - 2    

# Find end position character in validation data
for answer, text in zip(val_answers, val_texts):
    real_answer = answer['text']
    start_idx = answer['answer_start']
    # Get the real end index
    end_idx = start_idx + len(real_answer)

    # Deal with the problem of 1 or 2 more characters 
    if text[start_idx:end_idx] == real_answer:
        answer['answer_end'] = end_idx
    # When the real answer is more by one character
    elif text[start_idx-1:end_idx-1] == real_answer:
        answer['answer_start'] = start_idx - 1
        answer['answer_end'] = end_idx - 1  
    # When the real answer is more by two characters  
    elif text[start_idx-2:end_idx-2] == real_answer:
        answer['answer_start'] = start_idx - 2
        answer['answer_end'] = end_idx - 2
        
# Inicializamos Fabric
fabric = Fabric(accelerator="cuda", devices=2, num_nodes=2, strategy=args.strategy)
fabric.launch()


# Preprocesado de los datos para tokenizar 
# transformers.AdamW = torch.optim.AdamW
from transformers import AutoTokenizer,BertForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, train_queries, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, val_queries, truncation=True, padding=True)

# Convertir start-end position en start-end token position
def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []

  count = 0

  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

    # if start position is None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
      
    # if end position is None, the 'char_to_token' function points to the space after the correct token, so add - 1
    if end_positions[-1] is None:
      end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
      # if end position is still None the answer passage has been truncated
      if end_positions[-1] is None:
        count += 1
        end_positions[-1] = tokenizer.model_max_length


  # Update the data in dictionary
  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

# Create a Dataset class
# Create a Squatdataset class (inherits from torch.utils.data.Dataset), that helped me to train and validate my previous data more easily and convert encodings to datasets.

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)
    
# Use data loader
batch_size= 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Select GPU
# device = torch.device('cuda' if torch.cuda.is_available()
#                      else 'cpu')
# if device.type == 'cuda':
	# print(f"Usando la GPU: {torch.cuda.get_device_name(0)}")

# Build the BERT model. Optimizer, learning rate and epoch
lr=5e-5
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
optim = AdamW(model.parameters(), lr)
epochs = 5


# Setup con Fabric
model, optim = fabric.setup(model, optim)
train_loader = fabric.setup_dataloaders(train_loader)
val_loader = fabric.setup_dataloaders(val_loader)

# Train and Evaluate Model
whole_train_eval_time = time.time()

train_losses = []
val_losses = []

print_every = 1000

for epoch in range(epochs):
  epoch_time = time.time()

  # Set model in train mode
  model.train()
    
  loss_of_epoch = 0

  if fabric.is_global_zero:
      print("############Train############")

  for batch_idx,batch in enumerate(train_loader): 
    
    optim.zero_grad()

    # input_ids = batch['input_ids'].to(fabric.device)
    # attention_mask = batch['attention_mask'].to(fabric.device)
    # start_positions = batch['start_positions'].to(fabric.device)
    # end_positions = batch['end_positions'].to(fabric.device)

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    start_positions = batch['start_positions']
    end_positions = batch['end_positions']
    
    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
    loss = outputs.loss
    loss = loss.mean()
    # do a backwards pass 
    fabric.backward(loss)
    # update the weights
    optim.step()
    # Find the total loss
    loss_of_epoch += loss.item()

    if (batch_idx+1) % print_every == 0:
      print("Batch {:} / {:}".format(batch_idx+1,len(train_loader)),"\nLoss:", round(loss.item(),1),"\n")

  loss_of_epoch /= len(train_loader)
  train_losses.append(loss_of_epoch)

  ##########Evaluation##################

  # Set model in evaluation mode
  model.eval()
  if fabric.is_global_zero:
      print("############Evaluate############")

  loss_of_epoch = 0

  for batch_idx,batch in enumerate(val_loader):
    
    with torch.no_grad():

      # input_ids = batch['input_ids'].to(fabric.device)
      # attention_mask = batch['attention_mask'].to(fabric.device)
      # start_positions = batch['start_positions'].to(fabric.device)
      # end_positions = batch['end_positions'].to(fabric.device)

      input_ids = batch['input_ids']
      attention_mask = batch['attention_mask']
      start_positions = batch['start_positions']
      end_positions = batch['end_positions']    
      
      outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
      loss = outputs.loss
      loss = loss.mean()
      # Find the total loss
      loss_of_epoch += loss.item()

    if (batch_idx+1) % print_every == 0:
       print("Batch {:} / {:}".format(batch_idx+1,len(val_loader)),"\nLoss:", round(loss.item(),1),"\n")

  loss_of_epoch /= len(val_loader)
  val_losses.append(loss_of_epoch)

  # Print each epoch's time and train/val loss
  if fabric.is_global_zero:
      print("\n-------Epoch ", epoch+1,
        "-------"
        "\nTraining Loss:", train_losses[-1],
        "\nValidation Loss:", val_losses[-1],
        "\nTime: ",(time.time() - epoch_time),
        "\n-----------------------",
        "\n\n")

if fabric.is_global_zero:
    print("Total training and evaluation time: ", (time.time() - whole_train_eval_time))

# Save results and Save model
# torch.save(model,"results/finetunedmodel")
