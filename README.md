# `Canarim-7B-VestibulAide`

Para mais detalhes sobre o modelo, exemplos de teste de desempenho, conjunto de dados e processo de treinamento, visite: [**canar.im**](https://canar.im/) ou [**nlp.rocks**](https://nlp.rocks/).

## Descrição do modelo

`Canarim-7B-VestibulAide` é um modelo do tipo "decoder-only" com 7 bilhões de parâmetros, desenvolvido especialmente para lidar com perguntas, exercícios e respostas de questões de vestibulares brasileiros. O modelo foi ajustado para a **língua portuguesa** e tem como objetivo auxiliar estudantes na compreensão e resolução de questões complexas frequentemente presentes em vestibulares.

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

instruction = "Abaixo está uma pergunta de uma prova. Escreva uma resposta que solucione adequadamente a questão.\n"

# Questão retirada da prova de vestibular da UNESP 2023 (2º fase - questão 27)
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

Resposta do modelo:

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

## Conjunto de dados de treinamento

Os dados utilizados para treinar o modelo consistem em uma ampla variedade de provas de vestibulares brasileiros, abrangendo vários anos e edições de mais de 15 vestibulares distintos. Essa coleção diversificada de dados de diferentes fontes proporciona uma compreensão abrangente das características e complexidade das questões de vestibulares ao longo do tempo. Cada conjunto de dados foi cuidadosamente processado e incorporado ao treinamento do modelo, garantindo representatividade das questões encontradas em vestibulares reais.

| Universidade   | Ano do Vestibular                    | Quantidade de Questões |
| -------------- | ------------------------------------ | ---------------------- |
| Fatec          | 2023, 2020, 2019, 2019 (+ 23 provas) | 1253                   |
| Alberteinstein | 2023, 2019, 2016, 2016 (+ 5 provas)  | 385                    |
| Unifesp        | 2023, 2023, 2019, 2019 (+ 36 provas) | 1255                   |
| Famerp         | 2023, 2023, 2019, 2019 (+ 12 provas) | 659                    |
| Famema         | 2023, 2019, 2018, 2022 (+ 2 provas)  | 199                    |
| Unicamp        | 2023, 2023, 2023, 2021 (+ 74 provas) | 1637                   |
| Pasusp         | 2009                                 | 46                     |
| Fgv-sp         | 2020, 2019, 2019, 2018 (+ 57 provas) | 2699                   |
| Fmabc          | 2023, 2018, 2022, 2021 (+ 1 provas)  | 365                    |
| Mackenzie      | 2019, 2017, 2015, 2013 (+ 38 provas) | 1329                   |
| Insper         | 2015, 2014, 2014, 2016 (+ 1 provas)  | 127                    |
| Pucsp          | 2020, 2018, 2015, 2013 (+ 20 provas) | 1220                   |
| Fuvest         | 2011, 2011, 2011, 2009 (+ 78 provas) | 2059                   |
| Unip           | 2023, 2022                           | 90                     |
| Ita            | 2015, 2015, 2015, 2015 (+ 27 provas) | 748                    |
| Enem           | 2022, 2022, 2022, 2022 (+ 26 provas) | 2388                   |
| Santacasa      | 2023, 2023, 2019, 2019 (+ 8 provas)  | 532                    |
| Unesp          | 2002, 2002, 2012, 2010 (+ 17 provas) | 775                    |

Para ver a lista completa dos anos e provas utilizadas, acesse a [lista completa de vestibulares]().

## Desempenho

O desempenho do modelo foi avaliado de duas formas: uma relacionada à capacidade de sugerir a alternativa correta e outra à eficácia da resolução na ajuda ao estudante a encontrar a alternativa correta em questões de múltipla escolha.

### Sugestão de alternativa correta

A primeira métrica avalia a precisão do modelo na sugestão da alternativa correta em questões de múltipla escolha. Nesse contexto, o modelo é considerado bem-sucedido quando a alternativa segerida corresponde à alternativa correta da questão.

| Prova utilizada | Número total de questões analisadas | Acurácia |
| --------------- | ----------------------------------- | -------- |
| ENEM 2022       | 37                                  | 43 %     |
| FATEC 2023      | 11                                  | 27%      |

## Correspondência com a Resolução

A segunda métrica centra-se na correspondência entre a resolução gerada pelo modelo e a alternativa correta da questão. Isso inclui não apenas a resolução da questão, mas também a capacidade do modelo em fornecer explicações claras e úteis que permitam ao estudante compreender e chegar à alternativa correta. Essa métrica mede a eficácia da resolução gerada pelo modelo em auxiliar o estudante em todo o processo de compreensão e resolução da questão

| Prova utilizada | Número total de questões analisadas | Acurácia    |
| --------------- | ----------------------------------- | ----------- |
| ENEM 2022       | 77                                  | **75,32 %** |
| FATEC 2023      | 49                                  | 59,18 %     |

## Uso e Limitações

### Uso pretendido

Este modelo destina-se a estudantes que desejam aprimorar suas habilidades na resolução de questões de vestibulares. Pode ser utilizado das seguintes maneiras:

1. **Geração de Resoluções**: O modelo é capaz de gerar resoluções passo a passo para questões específicas, auxiliando os estudantes a compreender o processo de resolução e os conceitos envolvidos.

2. **Revisão e Estudo**: O modelo pode ser empregado para revisar e estudar diferentes tópicos abordados em questões de vestibulares, oferecendo explicações detalhadas quando necessário.

### Limitações

O modelo apresenta um desempenho considerável na geração de resoluções, fornecendo explicações detalhadas sobre o processo de resolução de questões de vestibulares. No entanto, é importante notar que, atualmente, ele pode apresentar algumas limitações na sugestão da alternativa correta em questões que envolvem a escolha entre várias opções.

A qualidade das sugestões para a alternativa correta pode nem sempre atender aos padrões desejados, podendo ocasionalmente resultar em respostas imprecisas ou inadequadas (mesmo quando a resolução/explicação está correta). É fundamental destacar que estou ciente dessas limitações e estou empenhado em aprimorar essa capacidade em versões futuras do modelo.

Estou comprometido em investir tempo e esforço para melhorar a qualidade das sugestões das alternativas corretas, mas isso pode levar algum tempo. Por enquanto, o modelo pode ser utilizado para gerar resoluções e explicações detalhadas, porém é recomendado que o usuário verifique a alternativa correta por conta própria.

Agradeço a compreensão de todos os usuários do modelo e valorizo o feedback de todos. Se você tiver alguma sugestão ou comentário, não hesite em entrar em contato comigo.

## Como citar

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
