# Vou construir um LLM do zero. Em público, e em dois idiomas.

*Série "LLM Bilíngue do Zero" · Artigo 0 · [Read in English](en.md)*

Em 2026, basta um comando para baixar um modelo de linguagem excelente e rodá-lo no notebook. Então por que alguém passaria seis meses construindo um modelo pior?

Porque usar não é entender. E eu quero entender.

## Por que do zero, e por que agora

Os modelos abertos de hoje são extraordinários. Dá para ajustar um deles com fine-tuning (reajuste de um modelo pronto com dados novos) numa tarde e ter algo útil no fim do dia. Só que o fine-tuning ensina a girar botões de uma máquina que continua fechada. Você aprende que funciona, mas não aprende *por que* funciona.

Construir do zero obriga a responder às perguntas que o atalho esconde:

- O que exatamente é um token, e por que o português custa mais tokens que o inglês?
- O que o modelo está "aprendendo" quando a loss (a medida de erro que o treino tenta reduzir) cai?
- O que o attention faz de verdade, sem metáforas vagas?
- Quando o treino dá errado, onde está o erro?

Tenho dez anos de carreira como desenvolvedor, a maior parte com Node.js. Coloquei em produção aplicações de back-end, front-end e mobile, e montei infraestruturas inteiras na AWS. Aprendi cada uma dessas coisas construindo, não lendo sobre elas. Vou aprender modelos de linguagem do mesmo jeito: escrevendo cada peça com as minhas mãos.

## O que vou construir

São doze missões em 25 semanas, de 28 de setembro de 2026 a março de 2027:

1. Um modelo que prevê a próxima letra contando pares de caracteres, em Python puro.
2. Um motor de gradientes (a regra que diz para onde ajustar cada número do modelo) escrito do zero.
3. A primeira rede neural em PyTorch.
4. Um tokenizer BPE, o mesmo tipo de algoritmo que corta o texto em pedaços nos modelos da família GPT.
5. Attention.
6. Um GPT completo, montado peça por peça e validado com os pesos oficiais do GPT-2.
7. Pretraining (o treino inicial, em que o modelo aprende a língua) num MacBook.
8. As melhorias de arquitetura que os modelos modernos usam, cada uma medida isoladamente.
9. Escala: dados reais, scaling laws e uma GPU alugada.
10. SFT (ajuste supervisionado com exemplos de conversa): transformar um completador de texto num assistente.
11. Lançamento do modelo e retrospectiva.

Até a missão 8, tudo roda num MacBook com chip M3 e 8 GB de memória. Essa restrição é de propósito: ela obriga a entender cada byte que o treino consome. Só na missão 9 entra uma GPU alugada, e o custo real vai ser publicado.

## A promessa para você

**LLMs explicados para quem programa.** Se você sabe o que é uma classe, uma interface e um objeto, tem a base necessária. Cada conceito novo vem com uma ponte para o que um desenvolvedor já conhece, e a matemática aparece no momento em que é necessária, nunca antes.

**Construído com TDD.** Cada componente é escrito com test-driven development: o teste vem antes do código. Um modelo construído do zero merece o mesmo rigor que qualquer outro software. Testar código de machine learning tem desafios próprios, como aleatoriedade e números aproximados, e eles também viram assunto da série.

**Tudo reproduzível.** Todo número publicado sai do código que está no repositório. Se eu afirmar que um tokenizer gasta 30% mais tokens em português, você pode rodar o experimento e conferir.

**Em dois idiomas.** Todo artigo sai em português e em inglês. O modelo final também vai ser bilíngue: material técnico de qualidade sobre LLMs em português ainda é raro, e um modelo pequeno que fala português é ainda mais raro.

**Aberto de verdade.** O código usa a licença MIT e os artigos, a CC BY 4.0. O modelo vai ser treinado só com dados de licença permissiva, para que qualquer pessoa possa usá-lo sem letra miúda.

## Como vai ser o sucesso, em março de 2027

Quero critérios que qualquer pessoa possa verificar, não impressões:

- As doze missões, todas construídas com TDD, num repositório público.
- Doze artigos em português e doze em inglês, além de uma entrega pública por semana.
- Um modelo bilíngue publicado no Hugging Face Hub, com model card (a ficha técnica do modelo), treinado só com dados de licença permissiva e avaliado nos dois idiomas.
- Um chat que roda na sua máquina.
- O custo total de treino publicado, até o último centavo.
- Um guia consolidado para quem quiser refazer o caminho.

## O que este modelo não vai ser

Prefiro alinhar as expectativas agora.

**Não vai competir com GPT, Claude, Llama ou Qwen.** O modelo final vai ter a escala do GPT-2: centenas de milhões de parâmetros, no máximo. Ele vai errar, inventar fatos e se perder em conversas longas. O objetivo não é superar ninguém. É entender cada decisão que o levou a funcionar, e a errar.

**Não é um produto.** É um modelo de estudo, documentado de ponta a ponta.

**Não é um curso dado por um especialista em machine learning.** Sou um engenheiro de software experiente aprendendo machine learning com rigor e em público. Todo artigo tem uma seção chamada "onde eu errei". É, provavelmente, a mais útil de todas.

## Quem escreve o quê

Todo o código deste projeto é meu, escrito com TDD, teste por teste. É ali que está o aprendizado, e é isso que quero demonstrar. Os artigos são escritos com apoio de IA, sempre a partir do meu código, dos meus números e dos meus erros. Nenhum resultado é inventado, e todos podem ser conferidos no repositório.

## Como acompanhar

Toda semana sai algo: código, um post curto de progresso ou um artigo. O repositório é [bilingual-llm-from-scratch](https://github.com/luizcampos331/bilingual-llm-from-scratch), e os artigos saem aqui no LinkedIn.

O próximo artigo começa pelo modelo de linguagem mais simples possível: contar quais letras costumam vir depois de quais. Sem redes neurais e sem bibliotecas, só Python e um dicionário. Vai ser surpreendente o quanto dá para entender com tão pouco.

---

*Código: [github.com/luizcampos331/bilingual-llm-from-scratch](https://github.com/luizcampos331/bilingual-llm-from-scratch) · [English version](en.md) · Texto sob CC BY 4.0*
