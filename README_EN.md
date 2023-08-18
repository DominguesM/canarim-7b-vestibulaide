# `Canarim-7B-VestibulAide`

For more details on the model, performance test examples, data set, and training process, visit: [**canar.im**](https://canar.im/) or [**nlp.rocks**](https://nlp.rocks/).

## Model Description

`Canarim-7B-VestibulAide` is a "decoder-only" model with 7 billion parameters, designed specifically for handling questions, exercises, and answers from Brazilian university entrance exams. Tailored for the **Portuguese language**, its aim is to assist students in understanding and solving complex questions commonly found in these exams.

## Usage

```python
from transformers import AutoTokenizer, pipeline
import torch

model_id = "dominguesm/canarim-7b-vestibulaide"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

instruction = "Abaixo está uma pergunta de uma prova. Escreva uma resposta que solucione adequadamente a questão.\n"

# Question from the 2023 UNESP University Entrance Exam (2nd phase - question 27)
question = """
Boric também disse que será duro com o narcotráfico e
com a delinquência, e que dará atenção às regiões que
estão em estado de exceção no norte e no sul do país.
“Mas nunca nos esqueçamos, por favor, que todos são
seres humanos”, referindo-se tanto a imigrantes ilegais
que vêm causando conflitos com os moradores locais, no
norte, como aos mapuche, “com quem temos dívidas
históricas”.
O presidente eleito pediu que o plebiscito que definirá a
implementação ou não de uma nova Constituição seja
“um grande momento de encontro”. [...]
Parafraseou algumas das últimas palavras de Salvador
Allende em seu último discurso antes de morrer,
afirmando que “seguiremos caminhando pelas grandes
alamedas por onde anda o homem livre para construir
uma sociedade melhor”.
   (Sylvia Colombo. “Boric toma posse como presidente do Chile e
         cita Allende em discurso”. Folha de S.Paulo, 11.03.2022.)


A matéria jornalística sobre a posse de Gabriel Boric na
presidência do Chile revela que o novo presidente
a) defende princípios da Constituição promulgada
   durante o regime militar chileno.
b) reafirma o mito da relação harmoniosa entre brancos e
   nativos sul-americanos.
c) endossa a intolerância de parte expressiva da
   população ao crescimento da imigração no país.
d) associa-se ao recente avanço eleitoral da direita latino-
   americana.
e) pretende recuperar parte da tradição histórica de luta
   dos socialistas chilenos.
"""

sequences = pipe(
    f"{instruction}###Pergunta\n{question}###Solução:\n",
    do_sample=True,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1024,
    temperature=0.9,
    top_p=0.6,
    repetition_penalty=1.15
)

```

Model's Answer:

```
Resolução:

O texto indica a admiração de Gabriel Boric pela
construção política realizada pelo socialista salvadorado
Salvador Allende, preso, assassinado e deposto no
Regime Militar apoiado pelos EUA, liderado pelo coronel
Pinochet. Assim, é possível concluir que o líder
atual retoma as ideias e perspectivas desenvolvidas sob
o governo anterior.
```

## Training Data Set

The data used to train the model consists of a wide range of Brazilian university entrance exams, spanning multiple years and editions from over 15 distinct entrance exams. This diverse data collection from different sources provides a comprehensive understanding of the characteristics and complexity of entrance exam questions over time. Each data set was meticulously processed and incorporated into the model's training, ensuring representation of questions found in actual entrance exams.

| University     | Exam Year                           | Number of Questions |
| -------------- | ----------------------------------- | ------------------- |
| Fatec          | 2023, 2020, 2019, 2019 (+ 23 exams) | 1253                |
| Alberteinstein | 2023, 2019, 2016, 2016 (+ 5 exams)  | 385                 |
| Unifesp        | 2023, 2023, 2019, 2019 (+ 36 exams) | 1255                |
| Famerp         | 2023, 2023, 2019, 2019 (+ 12 exams) | 659                 |
| Famema         | 2023, 2019, 2018, 2022 (+ 2 exams)  | 199                 |
| Unicamp        | 2023, 2023, 2023, 2021 (+ 74 exams) | 1637                |
| Pasusp         | 2009                                | 46                  |
| Fgv-sp         | 2020, 2019, 2019, 2018 (+ 57 exams) | 2699                |
| Fmabc          | 2023, 2018, 2022, 2021 (+ 1 exams)  | 365                 |
| Mackenzie      | 2019, 2017, 2015, 2013 (+ 38 exams) | 1329                |
| Insper         | 2015, 2014, 2014, 2016 (+ 1 exams)  | 127                 |
| Pucsp          | 2020, 2018, 2015, 2013 (+ 20 exams) | 1220                |
| Fuvest         | 2011, 2011, 2011, 2009 (+ 78 exams) | 2059                |
| Unip           | 2023, 2022                          | 90                  |
| Ita            | 2015, 2015, 2015, 2015 (+ 27 exams) | 748                 |
| Enem           | 2022, 2022, 2022, 2022 (+ 26 exams) | 2388                |
| Santacasa      | 2023, 2023, 2019, 2019 (+ 8 exams)  | 532                 |
| Unesp          | 2002, 2002, 2012, 2010 (+ 17 exams) | 775                 |

To view the complete list of years and exams used, access the [full list of entrance exams]().

## Performance

The model's performance was assessed in two ways: its ability to suggest the correct choice and its effectiveness in aiding students to identify the correct answer in multiple-choice questions.

### Correct Choice Suggestion

The first metric gauges the model's accuracy in suggesting the correct choice in multiple-choice queries. Here, the model is deemed successful when its suggested option matches the question's correct answer.

| Test Used  | Total Questions Reviewed | Accuracy |
| ---------- | ------------------------ | -------- |
| ENEM 2022  | 37                       | 43%      |
| FATEC 2023 | 11                       | 27%      |

## Solution Matching

The second metric focuses on the alignment between the model's generated solution and the question's correct choice. This covers not just the question's solution but also the model's capacity to provide clear, helpful explanations, allowing students to understand and pinpoint the correct answer. This metric measures the efficacy of the model's generated solution in assisting the student throughout the entire comprehension and problem-solving process.

| Test Used  | Total Questions Reviewed | Accuracy   |
| ---------- | ------------------------ | ---------- |
| ENEM 2022  | 77                       | **75.32%** |
| FATEC 2023 | 49                       | 59.18%     |

## Use and Limitations

### Intended Use

This model is intended for students aiming to enhance their skills in solving university entrance exam questions. It can be used in the following ways:

1. **Solution Generation**: The model can produce step-by-step solutions for specific questions, aiding students in grasping the solution process and the underlying concepts.

2. **Review and Study**: The model can be used for reviewing and studying various topics covered in entrance exam questions, providing detailed explanations when needed.

### Limitations

The model performs notably in generating solutions, offering in-depth explanations on the solution process of entrance exam questions. However, it's essential to point out that it may currently have some limitations in suggesting the correct option in multiple-choice questions.

The quality of suggestions for the right choice may not always meet the desired standards, possibly resulting in inaccurate or inappropriate answers (even when the solution/explanation is accurate). It's vital to stress that I'm aware of these limitations and am committed to enhancing this ability in future model versions.

I'm dedicated to investing time and effort to improve the quality of the correct option suggestions, but this might take a while. For now, the model can be used to generate solutions and in-depth explanations, but it's advised that users verify the right choice independently.

I appreciate the understanding of all model users and value everyone's feedback. If you have any suggestions or comments, please do not hesitate to reach out to me.

## How to Cite

```bibtex
@misc{Canarim7BVestibulAide,
    url    = {[https://huggingface.co/dominguesm/canarim-7b-vestibulaide](https://huggingface.co/dominguesm/canarim-7b-vestibulaide)},
    title  = {Canarim 7B VestibulAide},
    author = {DOMINGUES, M. F.}
}
```

## Citations

```bibtex
@misc{touvron2023llama,
      title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
      author={Hugo Touvron and Louis Martin and Kevin Stone and Peter Albert and Amjad Almahairi and Yasmine Babaei and Nikolay Bashlykov and Soumya Batra and Prajjwal Bhargava and Shruti Bhosale and Dan Bikel and Lukas Blecher and Cristian Canton Ferrer and Moya Chen and Guillem Cucurull and David Esiobu and Jude Fernandes and Jeremy Fu and Wenyin Fu and Brian Fuller and Cynthia Gao and Vedanuj Goswami and Naman Goyal and Anthony Hartshorn and Saghar Hosseini and Rui Hou and Hakan Inan and Marcin Kardas and Viktor Kerkez and Madian Khabsa and Isabel Kloumann and Artem Korenev and Punit Singh Koura and Marie-Anne Lachaux and Thibaut Lavril and Jenya Lee and Diana Liskovich and Yinghai Lu and Yuning Mao and Xavier Martinet and Todor Mihaylov and Pushkar Mishra and Igor Molybog and Yixin Nie and Andrew Poulton and Jeremy Reizenstein and Rashi Rungta and Kalyan Saladi and Alan Schelten and Ruan Silva and Eric Michael Smith and Ranjan Subramanian and Xiaoqing Ellen Tan and Binh Tang and Ross Taylor and Adina Williams and Jian Xiang Kuan and Puxin Xu and Zheng Yan and Iliyan Zarov and Yuchen Zhang and Angela Fan and Melanie Kambadur and Sharan Narang and Aurelien Rodriguez and Robert Stojnic and Sergey Edunov and Thomas Scialom},
      year={2023},
      eprint={2307.09288},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
