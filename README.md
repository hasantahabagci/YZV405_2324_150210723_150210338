# Natural Language Processing Term Project

This repository contains the code and documentation for our term project in the course titled "Natural Language Processing". Our project focuses on the diacritization of the Turkish language using a bi-directional Long Short-Term Memory (LSTM) network.

## Repository Structure

```bash
YZV405_2324_150210723_150210338/

├── data_loader/
│   ├── constantsForData.py
│   ├── data_loaders.py
│   └── utils.py
├── model/
│   ├── loss.py
│   ├── model.py
│   └── trainer.py
├── .gitignore
├── LICENSE
├── README.md
├── main.ipynb
├── playground.ipynb
├── requirements.txt
├── submission.csv
├── test.csv
├── train.csv
├── train_modified.csv
├── new_data.csv
├── news.csv
├── turkishaddresses.csv
├── turkishaddresses_indexed.csv
├── wiki.tr_indexed.txt
├── wiki.tr.txt
└── model_best.pth

```



## Introduction
In the realm of natural language processing (NLP), diacritization is a challenging task that involves adding diacritical marks to letters, which significantly impacts the meaning, pronunciation, and grammatical context of words in many languages. Our work explores innovative approaches to tackle the diacritization problem, particularly focusing on the Turkish language.

#### How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/hasantahabagci/YZV405_2324_150210723_150210338.git
    ```

2. Follow the instructions in `main.ipynb` to decide whether to use external datasets/our pre-trained model etc.

3. To try a sentence:
    - For simplicity, use `main.ipynb` and follow the steps to use the pre-trained model.
    - Example(Thanks  GPT-4o for this beautiful joke(!)):
        ```python
        sentence = "NLP modeli neden terapiye gitmis? Cunku cok fazla cozulmemis cumlesi varmıs!"
        result = try_a_sentence(sentence)          
        print(result)                              
        
        # output: NLP modeli neden terapiye gitmiş? Çünkü çok fazla çözülmemiş cümlesi varmış!
        ```

    

## Problem and Dataset
The diacritization of Turkish presents unique challenges due to the significant impact of diacritics on the semantic content of words. Our dataset comprises a collection of undiacritized Turkish text, posing challenges in identifying and correcting diacritic and character errors while preserving the intended meaning of words. The dataset was divided into train and test sets provided by our instructor from the ITU NLP Group. We also incorporated several external datasets to improve our model's performance.

## Model
The core of our project is a deep learning model designed to handle sequence data, specifically utilizing a bi-directional Long Short-Term Memory (LSTM) network. Implemented using PyTorch, this model leverages various neural network components to capture and predict patterns in sequential data.

### Model Architecture
- **Embedding Layer:** Transforms input tokens into dense vectors.
- **Bi-directional LSTM Layer:** Processes sequence data in both forward and backward directions to capture contextual information from both past and future tokens.
- **Dropout Layer:** Enhances generalization and prevents overfitting.
- **Fully Connected Layers:** Further process the hidden states to generate the final output.
- **Training:** Utilizes CrossEntropyLoss with label smoothing and an Adam optimizer with a learning rate of 0.01.
- 


## Results
Our results demonstrate the effectiveness of the bi-directional LSTM model in diacritizing Turkish text. The model achieved an accuracy of 96.622% on the public test set. Example diacritizations include:

- "Rusyadan gelen arkadasim dondu" → "Rusyadan gelen arkadaşım döndü"
- "Rusyadan gelen arkadasim donmus" → "Rusyadan gelen arkadaşım donmuş"
- "Muvaffakiyetsizlestiricilestiri veremeyebileceklerimizdenmissinizcesine" → "Muvaffakiyetsizleştiricileştiri veremeyebileceklerimizdenmişsinizcesine"
- "unlu yazar, donerek camdan dustu; babam, olmusla olmuse care yok dedi." → "ünlü yazar, dönerek camdan düştü; babam, olmuşla ölmüşe çare yok dedi."
- "NLP modeli neden bulmacayı cozmeyi birakti? cunku her cumlenin sonunu tahmin etmeye calisirken basi donmustu !" → "NLP modeli neden bulmacayı çözmeyi bıraktı? çünkü her cümlenin sonunu tahmin etmeye çalışırken başı dönmüştür"

## Conclusion
Our bi-directional LSTM model effectively addresses the diacritization challenge for Turkish, demonstrating high accuracy and computational efficiency. This project highlights the potential of deep learning models in solving complex NLP tasks.

## Authors
- **Muhammet Serdar NAZLI**  
  Istanbul Technical University  
  Faculty of Computer and Informatics  
  Artificial Intelligence and Data Engineering Department  
  [nazlim21@itu.edu.tr](mailto:nazlim21@itu.edu.tr)

- **Hasan Taha BAĞCI**  
  Istanbul Technical University  
  Faculty of Computer and Informatics  
  Artificial Intelligence and Data Engineering Department  
  [bagcih21@itu.edu.tr](mailto:bagcih21@itu.edu.tr)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
We thank our instructor and the ITU NLP Group for providing the dataset and guidance throughout the project.
