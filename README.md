# `Canarim-7B-VestibulAide`

Para mais detalhes sobre o modelo, exemplos de teste de desempenho, conjunto de dados e processo de treinamento, visite: [**canar.im**](https://canar.im/) ou [**nlp.rocks**](https://nlp.rocks/).

## Descrição do Modelo

`Canarim-7B-VestibulAide` é um modelo "apenas-decodificador" com 7 bilhões de parâmetros, projetado especificamente para lidar com perguntas, exercícios e respostas de vestibulares brasileiros. Adaptado para a **língua portuguesa**, seu objetivo é auxiliar os estudantes a compreender e resolver questões complexas comumente encontradas nesses exames.

## Uso

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

mensagem_sistema = """Você é um assistente prestativo, respeitoso e honesto, especializado na análise de questões de múltipla escolha, juntamente com a opção considerada correta que sempre responde as perguntas na língua portuguesa. Você oferece uma resposta abrangente, detalhada e bem fundamentada, explicando por que a opção escolhida é a correta. Garanta a abordagem de todos os aspectos relevantes da questão e forneça uma justificativa sólida que exponha com clareza por que a opção selecionada é a resposta correta. Suas resoluções visam proporcionar um entendimento completo da questão, permitindo que os leitores compreendam plenamente o raciocínio subjacente à resposta correta."""

def criar_prompt(instrução):
    return (
        f"[INST] <<SYS>>\n{mensagem_sistema.strip()}\n<</SYS>>\n\n"
        + instrução
        + " [/INST] "
    )

pergunta = """
É hoje a nossa festa nacional. O Brasil inteiro, da capital do Império à mais remota e insignificante de suas aldeolas, congrega-se unânime para comemorar o dia que o tirou dentre as nações dependentes para colocá-lo entre as nações soberanas, e entregou-lhes seus destinos, que até então haviam ficado a cargo de um povo estranho.

Gazeta de Notícias, 7 set.1883.

As festividades em torno da Independência do Brasil marcam o nosso calendário desde os anos imediatamente posteriores ao de setembro de 1822. Essa comemoração está diretamente relacionada com:

a) a construção e manutenção de símbolos para a formação de uma identidade nacional.
b) o domínio da elite brasileira sobre os principais cargos políticos, que se efetivou logo após 1822.
c) os interesses de senhores de terras que, após a Independência, exigiram a abolição da escravidão.
d) o apoio popular às medidas tomadas pelo governo imperial para a expulsão de estrangeiros do país.
e) a consciência da população sobre os seus direitos adquiridos posteriormente à transferência da Corte para o Rio de Janeiro.
"""

sequencias = pipe(
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

Resposta do Modelo:

```
Aqui está minha resposta à sua pergunta:
A opção correta é (a) a construção e manutenção de símbolos para a formação de uma identidade nacional.
A independência do Brasil foi um momento histórico fundamental para a formação da nação, pois representou a transição da fase colonial para a fase republicana. Para consolidar essa mudança e criar uma identificação nacional compartilhada, foram precisos símbolos e ativos culturais que expressassem a cultura, a história e os valores da nova nação. Assim, a construção e manutenção de símbolos como a Bandeira, o Hino Nacional, a Câmara Legislativa, entre outros, foi fundamental para a formação da identidade nacional brasileira. Além disso, esses símbolos também representam a liberdade e a soberania do país, o que é importante para a construção de uma nação forte e independente. Por isso, a opção (a) é a correta.
```

## Conjunto de Dados de Treinamento

Os dados usados para treinar o modelo consistem em uma ampla variedade de exames de vestibulares de universidades brasileiras e concursos públicos, abrangendo vários anos e edições de mais de 15 exames de vestibulares distintos e 50 concursos públicos. Essa coleção diversificada de dados de várias fontes proporciona uma compreensão abrangente das características e complexidade das questões de vestibulares ao longo do tempo. Cada conjunto de dados foi meticulosamente processado e incorporado ao treinamento do modelo, garantindo a representação de questões encontradas em exames de vestibulares reais.

Para ver a lista completa de anos e exames utilizados, acesse a [lista completa de exames de vestibulares](https://canar.im/).

## Desempenho

O desempenho do modelo foi avaliado de duas maneiras: sua capacidade de sugerir a escolha correta (ENEM 2022 - Dia 1) e métricas como **ROUGE** e **BLEU**.

### ENEM 2022 - Dia 1

A primeira métrica mede a precisão do modelo em sugerir a escolha correta em questões de múltipla escolha. Nesse caso, o modelo é considerado bem-sucedido quando a opção sugerida corresponde à resposta correta da questão.

O desempenho do modelo foi avaliado por meio do teste **ENEM 2022 - Dia 1**, que consiste em 84 questões de múltipla escolha. A precisão do modelo em sugerir a escolha correta foi de 35,71%, com 30 de 84 questões respondidas corretamente.

Os dados usados para derivar essa métrica podem ser encontrados em: [canarim-enem-22-test](https://huggingface.co/datasets/dominguesm/can

arim-enem2022-tests)

### ROUGE e BLEU

WIP

## Uso e Limitações

### Uso Pretendido

Este modelo destina-se a estudantes que visam aprimorar suas habilidades na resolução de questões de vestibulares universitários. Ele pode ser usado das seguintes maneiras:

1. **Geração de Soluções**: O modelo pode produzir soluções passo a passo para questões específicas, auxiliando os estudantes a compreender o processo de resolução e os conceitos subjacentes.

2. **Revisão e Estudo**: O modelo pode ser usado para revisar e estudar vários tópicos abordados em questões de vestibulares, fornecendo explicações detalhadas quando necessário.

### Limitações

O modelo se destaca na geração de soluções, oferecendo explicações detalhadas sobre o processo de solução de questões de vestibulares. No entanto, é importante ressaltar que ele pode atualmente ter algumas limitações em sugerir a opção correta em questões de múltipla escolha.

A qualidade das sugestões para a escolha correta nem sempre pode atender aos padrões desejados, possivelmente resultando em respostas imprecisas ou inadequadas (mesmo quando a solução/explicação é precisa). É fundamental enfatizar que estou ciente dessas limitações e estou comprometido em aprimorar essa habilidade em futuras versões do modelo.

Estou dedicado a investir tempo e esforço para melhorar a qualidade das sugestões para a opção correta, mas isso pode levar um tempo. Por enquanto, o modelo pode ser usado para gerar soluções e explicações detalhadas, mas é aconselhável que os usuários verifiquem a escolha correta de forma independente.

Agradeço a compreensão de todos os usuários do modelo e valorizo o feedback de todos. Se você tiver alguma sugestão ou comentário, não hesite em entrar em contato comigo.

## Como Citar

```bibtex
@misc{Canarim7BVestibulAide,
    url    = {[https://huggingface.co/dominguesm/canarim-7b-vestibulaide](https://huggingface.co/dominguesm/canarim-7b-vestibulaide)},
    title  = {Canarim 7B VestibulAide},
    author = {DOMINGUES, M. F.}
}
```

## Citações

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
