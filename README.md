# MOPPO: Aprendizaje por Refuerzo Multi-Objetivo para Anonimización de Grafos

**Proyecto final de Aprendizaje por Refuerzo**  
**Profesora:** Ileana Angélica Grave Aguilar  
**Autores:** Abril Minerva Estrada Montano y Sebastian Alanis González  
**Licenciatura en Ciencia de Datos**  
**Repositorio:** https://github.com/xaprilapril/RL-MOPPO  
**Año:** 2026

---

## Descripción general

Este repositorio contiene el desarrollo completo de **MOPPO**, un proyecto de aprendizaje por refuerzo multi-objetivo aplicado a la **anonimización de grafos sociales**. El problema central consiste en modificar una red para proteger la privacidad de sus nodos, pero sin destruir la estructura que hace útil al grafo para análisis posteriores.

En una red social, anonimizar no significa únicamente quitar nombres o identificadores. Aunque los atributos explícitos desaparezcan, la estructura del grafo puede seguir revelando información: el grado de un nodo, sus conexiones, su pertenencia a comunidades o su posición como hub/periferia pueden funcionar como huellas estructurales. Por ello, la anonimización de grafos exige modificar aristas, pero cada modificación altera propiedades importantes como grados, triángulos, clustering y conectividad.

Nuestro proyecto plantea esta tensión como un problema de **aprendizaje por refuerzo multi-objetivo**. En lugar de buscar una única solución fija, entrenamos y evaluamos métodos capaces de navegar el compromiso entre dos objetivos:

- **Privacidad:** medida mediante k-anonimato por grado.
- **Utilidad estructural:** medida mediante preservación del coeficiente de clustering y costo de modificación de aristas.

La idea principal es que no existe una única respuesta correcta para todos los contextos. En algunos escenarios puede ser aceptable sacrificar utilidad para obtener mayor privacidad; en otros, puede ser más importante conservar la estructura original del grafo. Por eso, MOPPO trabaja con una preferencia configurable representada por el parámetro `alpha`, que controla el peso relativo entre privacidad y utilidad.

---

## Motivación del proyecto

La anonimización de grafos es un problema especialmente delicado porque las redes no son bases de datos tabulares tradicionales. En una tabla, modificar un valor afecta una celda; en un grafo, modificar una arista afecta simultáneamente a dos nodos, puede cambiar sus grados, alterar clases de equivalencia, romper triángulos y modificar la interpretación de comunidades completas.

Este proyecto nace de una pregunta práctica:

> ¿Cómo podemos mejorar la privacidad de una red social sin perder por completo la información estructural que la hace útil?

Desde aprendizaje por refuerzo, esta pregunta se vuelve natural. Un agente puede observar el estado actual del grafo, decidir qué arista modificar, recibir una recompensa y continuar actuando hasta construir una versión anonimizada. Sin embargo, la dificultad real aparece porque la recompensa no es única: privacidad y utilidad pueden entrar en conflicto.

Por esa razón usamos un enfoque **multi-objetivo**, donde el agente no aprende solamente a maximizar una recompensa, sino a comportarse de forma distinta según la preferencia entre privacidad y utilidad.

---

## Objetivos

### Objetivo general

Formular, implementar y analizar un enfoque de aprendizaje por refuerzo multi-objetivo para anonimización de grafos, comparando MOPPO contra baselines clásicos, heurísticos y aleatorios bajo métricas de privacidad, utilidad, costo estructural, estabilidad y cobertura del frente de Pareto.

### Objetivos específicos

1. Formular la anonimización de grafos como un **MDP multi-objetivo**.
2. Implementar un ambiente `GraphAnonEnv` donde las acciones correspondan a modificaciones de aristas.
3. Entrenar un agente PPO condicionado por pesos para adaptarse a distintas preferencias `alpha`.
4. Incorporar **action masking** para reducir el espacio efectivo de acciones.
5. Usar **curriculum learning** para estabilizar el entrenamiento.
6. Evaluar el desempeño mediante privacidad, utilidad, score ponderado, hipervolumen y puntos Pareto.
7. Comparar MOPPO contra baselines:
   - Fixed-alpha PPO.
   - Random Masked.
   - Greedy Liu-Terzi style.
   - Alpha-Greedy Rollout.
8. Analizar la estabilidad por semillas y sensibilidad al valor de `alpha`.
9. Estudiar el costo estructural de anonimización y la preservación de clustering.
10. Documentar resultados en un reporte extendido, un paper base y notebooks experimentales.

---

## Idea central

MOPPO aprende una política condicionada por preferencias. Esto significa que el agente no recibe únicamente el estado del grafo, sino también un vector que indica qué tan importante es la privacidad frente a la utilidad:

```text
[alpha, 1 - alpha]
