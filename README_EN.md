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

system_message = """Você é um assistente prestativo, respeitoso e honesto, especializado na análise de questões de múltipla escolha, juntamente com a opção considerada correta que sempre responde as perguntas na lingua portugues (portugues). Você oferece uma resposta abrangente, detalhada e bem fundamentada, explicando por que a opção escolhida é a correta. Garanta a abordagem de todos os aspectos relevantes da questão e forneça uma justificativa sólida que exponha com clareza por que a opção selecionada é a resposta correta. Suas resoluções visam proporcionar um entendimento completo da questão, permitindo que os leitores compreendam plenamente o raciocínio subjacente à resposta correta."""

def make_prompt(instruction):
    return (
        f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n"
        + instruction
        + " [/INST] "
    )

question = """
É hoje a nossa festa nacional. O Brasil inteiro, da capital do Império à mais remota e insignificante de suas aldeolas, congrega-se unânime para comemorar o dia que o tirou dentre as nações dependentes para colocá-lo entre as nações soberanas, e entregou-lhes seus destinos, que até então haviam ficado a cargo de um povo estranho.

Gazeta de Notícias, 7 set.1883.

As festividades em torno da Independência do Brasil marcam o nosso calendário desde os anos imediatamente posteriores ao de setembro de 1822. Essa comemoração está diretamente relacionada com:

a) a construção e manutenção de símbolos para a formação de uma identidade nacional.
b) o domínio da elite brasileira sobre os principais cargos políticos, que se efetivou logo após 1822.
c) os interesses de senhores de terras que, após a Independência, exigiram a abolição da escravidão.
d) o apoio popular às medidas tomadas pelo governo imperial para a expulsão de estrangeiros do país.
e) a consciência da população sobre os seus direitos adquiridos posteriormente à transferência da Corte para o Rio de Janeiro.
"""

sequences = pipe(
   prompt,
   do_sample=True,
   num_return_sequences=1,
   eos_token_id=tokenizer.eos_token_id,
   max_length=2048,
   temperature=0.9,
   top_p=0.6,
   repetition_penalty=1.15
)

```

Model's Answer:

```
Aqui está minha resposta à sua pergunta:
A opção correta é (a) a construção e manutenção de símbolos para a formação de uma identidade nacional.
A independência do Brasil foi um momento histórico fundamental para a formação da nação, pois representou a transição da fase colonial para a fase republicana. Para consolidar essa mudança e criar uma identificação nacional compartilhada, foram precisos símbolos e ativos culturais que expressassem a cultura, a história e as valores da nova nação. Assim, a construção e manutenção de símbolos como a Bandeira, o Hino Nacional, a Câmara Legislativa, entre outros, foi fundamental para a formação da identidade nacional brasileira. Além disso, esses símbolos também representam a liberdade e a soberania do país, o que é importante para a construção de uma nação forte e independente. Por isso, a opção (a) é a correta.
```

## Training Data Set

The data used to train the model consists of a wide range of entrance exams from Brazilian universities and public competitions, spanning multiple years and editions of over 15 distinct entrance exams and 50 public contests. This diversified collection of data from various sources provides a comprehensive understanding of the characteristics and complexity of entrance exam questions over time. Each dataset was meticulously processed and incorporated into the model's training, ensuring the representation of questions found in actual entrance exams.

To view the complete list of years and exams used, access the [full list of entrance exams](https://canar.im/).

## Performance

The model's performance was evaluated in two ways: its ability to suggest the correct choice (ENEM 2022 - Day 1) and metrics such as **ROUGE** and **BLEU**.

### ENEM 2022 - Day 1

The first metric measures the model's accuracy in suggesting the correct choice in multiple-choice queries. Here, the model is deemed successful when the suggested option corresponds to the correct answer to the question.

The model's performance was evaluated through the **ENEM 2022 - Day 1** test, which consists of 84 multiple-choice questions. The model's accuracy in suggesting the correct choice was 35.71%, with 30 out of 84 questions answered correctly.

The data used to derive this metric can be found at: [canarim-enem-22-test](https://huggingface.co/datasets/dominguesm/canarim-enem2022-tests)

### ROUGE and BLEU

WIP

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
