# Resumen del Proyecto

Este proyecto de tesis explora la intersección entre el Aprendizaje Automático (Machine Learning) y la Inferencia Causal, centrándose específicamente en la aplicación del marco de "Double Machine Learning" (DML) en diseños de Diferencias en Diferencias (DiD). Tradicionalmente, ML se ha asociado con la predicción, mientras que las ciencias sociales buscan explicaciones causales. DML emerge como un método robusto para aprovechar la potencia predictiva de ML manteniendo la validez estadística para la estimación causal. Se examinan los fundamentos y se aplican a dos casos: uno canónico y otro de adopción escalonada, proporcionando una guía práctica para su implementación.

<!---
De una extensión de no más de 250 palabras deberá indicar: el tema del proyecto, su relevancia, los objetivos, su valor /originalidad y la metodología propuesta.
-->



# Justificación

Existe una dicotomía histórica en estadística aplicada entre predecir y explicar, tal como lo destacó Breiman (2001) en su influyente trabajo sobre las "dos culturas" del modelado estadístico. Por un lado, la cultura del modelado de datos, predominante en la econometría tradicional, asume que los datos son generados por un modelo estocástico conocido (e.g., regresión lineal) y se enfoca en la inferencia de parámetros y la causalidad. Por otro lado, la cultura del modelado algorítmico, base del Aprendizaje Automático (ML), trata el mecanismo generador de datos como desconocido y complejo, enfocándose casi exclusivamente en la precisión predictiva.

Los métodos de ML, como los bosques aleatorios o las redes neuronales, han demostrado ser excelentes para predecir ($\hat{Y}$), capturando relaciones no lineales complejas e interacciones de alto orden que los modelos tradicionales pasan por alto. Sin embargo, a menudo son "cajas negras" difíciles de interpretar causalmente, lo que ha generado escepticismo entre economistas y científicos sociales cuyo objetivo es entender *por qué* ocurre un fenómeno (explicar) y cómo una intervención afecta un resultado (inferencia causal).

Por otro lado, los métodos econométricos tradicionales son interpretables pero pueden sufrir de sesgos severos por "mala especificación del modelo" al tener que asumir una forma funcional rígida (e.g., linealidad, aditividad). Si la verdadera relación entre las variables de confusión ($X$) y el resultado ($Y$) o el tratamiento ($D$) e es compleja, un modelo lineal simple no logrará controlar adecuadamente, resultando en estimaciones sesgadas del efecto causal ("sesgo de variable omitida" o "sesgo de forma funcional").

La justificación de este trabajo radica en la necesidad imperiosa de herramientas que combinen lo mejor de ambos mundos: la flexibilidad y poder predictivo del ML para modelar las variables de confusión ("nuisance") y el rigor de la teoría de inferencia causal para obtener estimadores insesgados y realizar pruebas de hipótesis válidas. El marco de Double Machine Learning (DML), propuesto por Chernozhukov et al. (2018), ofrece una solución teóricamente robusta a este problema.

Este trabajo es relevante y oportuno porque proporciona evidencia empírica y una guía metodológica práctica sobre el uso de DML específicamente en diseños de Diferencias en Diferencias (DiD). DiD es una de las herramientas más populares en evaluación de políticas públicas, pero también una donde los supuestos tradicionales están siendo cuestionados. Mostrar cómo DML puede robustecer estos análisis tiene un valor directo para investigadores y profesionales en economía, ciencia de datos y políticas públicas.

<!---
En este punto corresponde argumentar sobre los motivos que determinaron la elección del tema a tratar. Se debe hacer referencia a la pertinencia y relevancia del tema dentro del área de conocimiento que aborda la carrera cursada.  Además, se puede dar cuenta de la elección personal del tema y sus antecedentes.
-->




# Planteamiento del tema/problema

El problema central de investigación es comparar los resultados de estimadores tradicionales (paramétricos) frente a estimadores basados en Machine Learning (semiparamétricos) en el contexto de inferencia causal, identificando escenarios donde no hay discrepancias significativas y, crucialmente, escenarios donde sí las hay.

En la práctica econométrica actual, el diseño de Diferencias en Diferencias (DiD) es el estándar para estimar efectos causales con datos de panel observacionales. El estimador convencional para implementarlo ha sido la regresión de Efectos Fijos de Dos Vías (Two-Way Fixed Effects - TWFE).
Sin embargo, una literatura reciente y creciente (e.g., Goodman-Bacon, 2021; Callaway & Sant'Anna, 2021) ha demostrado que TWFE puede presentar sesgos significativos, especialmente en configuraciones de adopción escalonada (*staggered adoption*), donde las unidades reciben el tratamiento en diferentes momentos. En estos casos, TWFE realiza comparaciones "prohibidas" (e.g., usar unidades tratadas temprano como control para unidades tratadas tarde), lo que puede resultar en ponderaciones negativas y estimaciones del efecto del tratamiento que tienen el signo opuesto al verdadero efecto.

Además del problema temporal, existe el problema de las covariables. TWFE asume típicamente que las covariables entran en el modelo de forma lineal y aditiva. Si la realidad es que las covariables interactúan entre sí de formas complejas o tienen efectos no lineales, TWFE estará mal especificado. Aquí es donde DML promete una mejora sustancial: al utilizar algoritmos de aprendizaje automático para aprender la forma de $E[Y|X]$ y $E[D|X]$ de los datos, DML puede "limpiar" el efecto de las covariables de manera mucho más efectiva que una regresión lineal, sin requerir que el investigador conozca la forma funcional exacta *a priori*.

La investigación aborda este problema aplicando el marco DML para flexibilizar los supuestos sobre las variables de control y corregir sesgos potenciales. No se trata simplemente de aplicar un algoritmo más complejo, sino de evaluar si esta complejidad adicional se traduce en una mejor inferencia estadística en casos reales.

Por lo tanto, se plantea la siguiente pregunta guía: **¿Cuándo el DML y TWFE presentan diferencias sustanciales en diseños de Diferencias en Diferencias?**
Específicamente:
*   ¿Son comparables los resultados en escenarios "canónicos" de tratamiento simultáneo y pocas covariables?
*   ¿Puede DML recuperar efectos causales en escenarios de adopción escalonada donde TWFE falla o subestima el efecto?
*   ¿Qué ventaja aporta la flexibilidad del ML en la modelización de tendencias contrafactuales condicionadas a covariables?

<!---
Plantear el tema/problema del TFM significa enunciar lo que se pretende investigar. Una buena formulación del tema delimita la investigación y sirve de guía. Es necesario explicitar los aspectos, factores o elementos relevantes relacionados con el problema que se va a investigar. Es conveniente, en este punto, plantear la/s pregunta/s problematizante/s que dará/n lugar a la formulación de los objetivos del trabajo.

El tema/problema del TFM debe:
•	Ser planteado en forma clara y precisa, con la mayor especificidad que sea posible establecer.
•	Contener la posibilidad de contrastación empírica y/o demostración lógica.
•	Estar delimitado en espacio y tiempo. 

-->

# Objetivos

# Objetivos

**Objetivo General:**

Analizar comparativamente y validar la eficacia de los estimadores de Double Machine Learning (DML) aplicados a diseños de Diferencias en Diferencias (DiD), con el fin de proporcionar a los investigadores aplicados una guía metodológica sobre cuándo y cómo utilizar estas técnicas avanzadas frente a los métodos econométricos tradicionales.

**Objetivos Específicos:**

1.  **Fundamentación Teórica:**
    
2.  **Implementación Metodológica:**
    *   Desarrollar e implementar computacionalmente (en Python) los algoritmos de estimación DML para DiD propuestos por Chang (2020) para tratamientos simultáneos.
    *   Implementar los estimadores doblemente robustos de Callaway & Sant'Anna (2021) adaptados con aprendizaje automático para tratamientos escalonados.

3.  **Evaluación Empírica:**
    *   Replicar el estudio de *Fracking* (tratamiento único) para testear la hipótesis de equivalencia entre DML y TWFE en escenarios simples.
    *   Reanalizar el caso de *Castle Doctrine* (tratamiento escalonado) para evaluar la capacidad de DML de corregir sesgos y detectar efectos heterogéneos que TWFE no captura.


<!---
En este punto, se enuncian los objetivos:

- Objetivo general: Orienta la investigación y es una referencia al trabajo a realizar. El objetivo general indica el resultado a alcanzar. Es aquel que brindará respuesta al problema/tema formulado. Es fundamental mantener la coherencia entre el tema/problema formulado y el objetivo general. Habitualmente, el objetivo general tiene un alto nivel de abstracción. Los objetivos se redactan a partir de un verbo en infinitivo, ya que expresan una acción a realizar (describir, explorar, analizar, determinar, establecer, desarrollar, diseñar, etc.).

- Objetivos específicos: precisan los requerimientos o propósitos con relación a la naturaleza del tema y están orientados por el objetivo general. Se desarrollan a partir de la operacionalización del objetivo general. Describen con mayor precisión  los productos a obtener y las variables/aspectos relevantes a estudiar para dar respuesta al tema/problema formulado. Es fundamental que dichos objetivos guarden correspondencia con el objetivo general del trabajo. El cumplimiento de cada objetivo específico constituiría por lo menos un capítulo del desarrollo del TFM.

-->


# Hipótesis

# Hipótesis

La investigación se guía por las siguientes hipótesis de trabajo, diferenciadas por el tipo de diseño cuasi-experimental:

**Hipótesis 1 (Escenarios Simples / Canónicos):**
En diseños de Diferencias en Diferencias con *tratamiento simultáneo* (todas las unidades tratadas comienzan en la misma fecha) y un número moderado de covariables lineales, los estimadores DML y TWFE convergerán a resultados estadísticamente indistinguibles.
*   *Razonamiento:* En estos casos, el estimador TWFE es insesgado bajo tendencias paralelas básicas, y la especificación lineal suele ser una aproximación suficiente. La complejidad adicional del DML no aportará una reducción significativa del sesgo.

**Hipótesis 2 (Escenarios Complejos / Adopción Escalonada):**
En diseños de Adopción Escalonada (*Staggered Aoption*), especialmente aquellos con heterogeneidad en el efecto del tratamiento, el estimador TWFE presentará sesgos significativos (potencialmente de signo opuesto), mientras que el estimador DML recuperará el efecto causal promedio correcto (ATT).
*   *Razonamiento:* DML, al utilizar grupos de control "limpios" (nunca tratados o aún no tratados) y ponderación por propensity score, evita las "comparaciones prohibidas" que contaminan a TWFE (Bacon, 2021).

**Hipótesis 3 (Flexibilidad Funcional):**
DML demostrará una mayor robustez y precisión (menores errores estándar o intervalos de confianza más creíbles) al controlar por covariables que tienen relaciones no lineales con el resultado.
*   *Razonamiento:* Al no imponer una forma funcional rígida, los algoritmos de ML capturarán mejor la varianza explicada por las variables de confusión, reduciendo la varianza residual y eliminando el sesgo de mala especificación.

<!---
Son anticipos de respuesta al problema planteado. Las hipótesis pueden elaborarse para guiar el trabajo de investigación (puede ser solo una presunción) y/o para plantear su verificación. Esto está en concordancia con los objetivos y tipo de estudio a realizar.

-->


# Marco teórico (preliminar)

El marco teórico de esta tesis integra dos literaturas que tradicionalmente han evolucionado por separado: la econometría de evaluación de impacto (específicamente Diferencias en Diferencias) y el aprendizaje automático estadístico (Double Machine Learning).

## 1. Diferencias en Diferencias (DiD)

El método de Diferencias en Diferencias es una de las estrategias de identificación más robustas y ampliamente utilizadas en economía aplicada para estimar efectos causales cuando no es posible realizar un experimento aleatorizado.

### 1.1 El Estimador Canónico
En su forma más simple (2 periodos, 2 grupos), el estimador DiD compara el cambio temporal en el grupo de tratamiento con el cambio temporal en el grupo de control. Formalmente, si $Y_{it}$ es el resultado para la unidad $i$ en el tiempo $t$, el estimador se define como:
$$ \hat{\delta}_{DiD} = (\bar{Y}_{post}^{Tratado} - \bar{Y}_{pre}^{Tratado}) - (\bar{Y}_{post}^{Control} - \bar{Y}_{pre}^{Control}) $$

Este "doble diferencial" elimina dos tipos de sesgos potenciales:
1.  Diferencias permanentes entre los dos grupos (efectos fijos de unidad).
2.  Tendencias temporales comunes que afectan a ambos grupos por igual (efectos fijos de tiempo).

Bajo ciertas condiciones, este estimador recupera el Efecto Promedio del Tratamiento en los Tratados (ATT), definido como $E[Y^1 - Y^0 | D=1]$, donde $Y^1$ y $Y^0$ son los resultados potenciales con y sin tratamiento, respectivamente.

### 1.2 Supuesto de Tendencias Paralelas
La validez del estimador DiD descansa críticamente en el **supuesto de tendencias paralelas**. Este supuesto establece que, en ausencia de tratamiento, la evolución promedio del resultado habría sido idéntica para ambos grupos. Matemáticamente:
$$ E[Y_t^0 - Y_{t-1}^0 | D=1] = E[Y_t^0 - Y_{t-1}^0 | D=0] $$
Si este supuesto se viola, el cambio observado en el grupo de control no es un contrafactual válido para el grupo de tratamiento, y el estimador DiD estará sesgado.

## 2. Limitaciones del DiD Tradicional

A pesar de su popularidad, la implementación estándar del DiD mediante regresión de mínimos cuadrados ordinarios (TWFE) enfrenta desafíos importantes que han sido destacados por la literatura econométrica reciente.

### 2.1 Adopción Escalonada (Staggered Adoption)
En muchas aplicaciones modernas, el tratamiento no se implementa al mismo tiempo para todas las unidades, sino que se adopta de forma escalonada. Goodman-Bacon (2021) demostró que el estimador TWFE estándar en estos contextos es una media ponderada de todas las posibles comparaciones de 2x2. Problemáticamente, algunas de estas ponderaciones pueden ser negativas, especialmente cuando los efectos del tratamiento varían en el tiempo. Esto puede llevar a situaciones paradójicas donde el estimador TWFE tiene el signo opuesto al verdadero efecto promedio.

### 2.2 La necesidad de Covariables
A menudo, el supuesto de tendencias paralelas incondicional es inverosímil. Por ejemplo, estados con diferentes características demográficas pueden tener tendencias de criminalidad naturales distintas. En estos casos, es necesario asumir **tendencias paralelas condicionales**: las tendencias son iguales *dado un vector de covariables observable $X$*.
La práctica estándar es incluir $X$ linealmente en la regresión ($Y \sim D + X$). Sin embargo, esto asume que sabemos exactamente cómo $X$ afecta a $Y$ (forma funcional). Si la relación es no lineal o implica interacciones complejas, el modelo lineal está mal especificado, reintroduciendo sesgo.

## 3. Double Machine Learning (DML)

Double Machine Learning, introducido por Chernozhukov et al. (2018), es un marco general para estimar parámetros causales estructurales en presencia de variables de confusión de alta dimensión ("nuisance parameters") que deben ser controladas de manera flexible.

### 3.1 Modelo Parcialmente Lineal
El punto de partida es a menudo un modelo parcialmente lineal (PLM):
$$ Y = D\theta_0 + g_0(X) + \zeta $$
$$ D = m_0(X) + \nu $$
Donde $Y$ es el resultado, $D$ es el tratamiento, y $\theta_0$ es el parámetro causal de interés. Crucialmente, $g_0(X)$ (la relación entre covariables y resultado) y $m_0(X)$ (la relación entre covariables y tratamiento, o *propensity score*) son funciones desconocidas y potencialmente muy complejas.

### 3.2 El problema de la Regularización
Si intentáramos estimar $\theta_0$ y $g_0(X)$ simultáneamente usando ML (e.g., Random Forest), el sesgo de regularización inherente al ML (necesario para evitar overfitting) se "contagiaría" a la estimación de $\theta_0$, resultando en un estimador sesgado y una inferencia inválida.

### 3.3 Ortogonalización de Neyman
DML resuelve esto mediante un proceso de dos etapas basado en el teorema de Frisch-Waugh-Lovell "generalizado". La idea es "ortogonalizar" el problema:
1.  Predecir $Y$ usando $X$ y obtener los residuos: $\tilde{Y} = Y - \hat{E}[Y|X]$.
2.  Predecir $D$ usando $X$ y obtener los residuos: $\tilde{D} = D - \hat{E}[D|X]$.
3.  Regresar $\tilde{Y}$ sobre $\tilde{D}$ para estimar $\theta_0$.

Esta estructura satisface la condición de **Ortogonalidad de Neyman**, lo que significa que el estimador de $\theta_0$ es localmente insensible a pequeños errores en la estimación de las funciones molestas $g_0$ y $m_0$. Esto permite usar algoritmos de ML ("cajas negras") para estimar $g_0$ y $m_0$ y aún así obtener inferencia válida de $\sqrt{N}$-consistente para $\theta_0$.

### 3.4 Cross-Fitting
Para evitar otro sesgo conocido como "overfitting bias" (usar los mismos datos para seleccionar el modelo y estimar el efecto), DML emplea **Cross-Fitting**:
1.  Dividir la muestra en $K$ pliegues ($folds$).
2.  Para cada pliegue $k$, usar el resto de los datos (complemento) para entrenar los modelos de ML.
3.  Usar los modelos entrenados para generar residuos en el pliegue $k$.
4.  Promediar los resultados.
Esto rompe la correlación espuria entre las estimaciones de las funciones molestas y los términos de error.

## 4. DML aplicado a Diferencias en Diferencias

Recientemente, autores como Chang (2020) y Callaway & Sant'Anna han extendido los principios de DML a configuraciones de panel y DiD.
En este contexto, DML se utiliza para estimar flexiblemente dos componentes clave:
1.  **Modelo de Resultado:** Predice la evolución del resultado para el grupo de control ($g(X)$), sirviendo como base contrafactual.
2.  **Modelo de Propensión:** Estima la probabilidad de ser tratado dado $X$ ($m(X)$), utilizado para reponderar el grupo de control y hacerlo comparable al tratado.

El estimador resultante es "doblemente robusto": es consistente si *o bien* el modelo de resultado *o bien* el modelo de propensión están correctamente especificados, y es eficientemente óptimo cuando ambos lo están. Esto contrasta con TWFE, que requiere que la especificación lineal sea correcta.

<!--
Constituye la enunciación de las referencias conceptuales que ayudan a comprender y focalizar el tema a estudiar. El marco teórico permite fundamentar el tema/problema a investigar y contextualizarlo. Implica la revisión de la literatura sobre el tema (estado del arte) y la identificación de los aspectos relevantes a tomar en cuenta para abordarlo.

A partir de esta revisión el maestrando realizará una selección de la literatura en función de sus objetivos. Su construcción supone un trabajo de análisis y toma de posición  por parte del maestrando. Se sugiere que la presentación del marco teórico (preliminar) se encuentre ordenada por subtítulos que faciliten su lectura y comprensión.  El marco teórico supone la referencia en el cuerpo del texto de los autores analizados para su construcción. Se utilizarán las normas APA en su edición más reciente para las referencias (autor/fecha) (consultar http://www.apastyle.org/)  

-->



# Metodología y técnicas a utilizar

Esta investigación adopta un enfoque **cuantitativo** con un diseño **observacional cuasi-experimental**. Se empleará una estrategia comparativa sistemática, aplicando tanto estimadores tradicionales (TWFE) como estimadores avanzados (DML) a los mismos conjuntos de datos para evaluar discrepancias.

## 1. Casos de Estudio y Fuentes de Datos

Se han seleccionado dos casos empíricos que representan distintos niveles de complejidad en el diseño DiD:

### Caso 1: Regulación Ambiental y Fracking (Diseño Canónico)
*   **Contexto:** Estimación del impacto de la expansión del *fracking* sobre la actividad regulatoria ambiental en Estados Unidos.
*   **Diseño:** Tratamiento simultáneo (un solo periodo de inicio del tratamiento en 2005) con grupos de tratamiento y control definidos geográficamente.
*   **Datos:** Panel balanceado a nivel de código postal-año con **143,275 observaciones**.
*   **Variables:**
    *   *Resultado ($Y$):* Número de acciones regulatorias, inspecciones y multas.
    *   *Tratamiento ($D$):* Indicador de exposición a pozos de fracking activos.
    *   *Covariables ($X$):* Empleo local, número de establecimientos comerciales (proxies de actividad económica).
*   **Hipótesis:** Dado que es un diseño canónico simple con pocas covariables, se espera que TWFE y DML arrojen resultados similares.

### Caso 2: Leyes "Castle Doctrine" y Homicidios (Diseño Staggered)
*   **Contexto:** Evaluación del efecto de las leyes de defensa propia extendida ("Castle Doctrine") sobre las tasas de homicidio estatales.
*   **Diseño:** Adopción escalonada (*staggered adoption*), donde diferentes estados aprobaron la ley en diferentes años entre 2000 y 2010.
*   **Datos:** Panel anual a nivel estatal cubriendo los 50 estados de EE.UU.
*   **Variables:**
    *   *Resultado ($Y$):* Tasa de homicidios logarítmica.
    *   *Tratamiento ($D$):* Indicador binario de ley vigente en el estado-año.
    *   *Covariables ($X$):* Población, ingresos, tasas de encarcelamiento, demografía (variables que evolucionan y pueden interactuar de forma compleja).
*   **Hipótesis:** Debido a la adopción escalonada y la heterogeneidad potencial entre estados, se espera que TWFE esté sesgado y que DML revele efectos distintos (posiblemente corrigiendo la subestimación o cambio de signo).

## 2. Técnicas de Procesamiento y Algoritmos

### Implementación de Machine Learning
Para la estimación de las funciones molestas ($g(X)$ y $m(X)$) en el marco DML, se utilizarán algoritmos basados en árboles de decisión, elegidos por su capacidad para manejar interacciones automáticas y no linealidades sin preprocesamiento exhaustivo:
*   **LightGBM:** Utilizado para el caso de Fracking debido a su alta eficiencia computacional con grandes volúmenes de datos (140k+ obs).
*   **Random Forest:** Utilizado para el caso de Castle Doctrine, dado que es robusto y tiende a funcionar bien con datos tabulares densos y muestras más pequeñas pero complejas.

### Estrategia de Validación
La validez interna de los resultados se evaluará mediante:
1.  **Estudios de Eventos (Event Studies):** Estimación de coeficientes pre-tratamiento (lags). Si los coeficientes para $t < 0$ son estadísticamente significativos, sugeriría una violación del supuesto de tendencias paralelas (i.e., los grupos ya divergían antes de la ley).
2.  **Pruebas de Placebo:** Ejecución de modelos con "tratamientos falsos" asignados aleatoriamente para verificar que el método no detecta efectos donde no existen.
3.  **Robustez de Semilla:** Dado que los algoritmos de ML tienen un componente estocástico, se ejecutarán múltiples iteraciones con diferentes semillas aleatorias para reportar la estabilidad de los estimadores e intervalos de confianza.

### Herramientas Computacionales
Todo el análisis se realizará en **Python**. Se utilizarán bibliotecas estándar (`pandas`, `numpy`) para manipulación de datos, `statsmodels` para estimaciones econométricas tradicionales, y herramientas específicas de DML o implementaciones personalizadas de los estimadores de Chang (2020) y Callaway & Sant'Anna (2021) integradas con `scikit-learn`.

<!---

<!--
En este punto, se enunciarán las decisiones tomadas y las tareas planificadas para desarrollar el trabajo de investigación. Se deberá enunciar el tipo de estudio previsto (enfoque cualitativo /cuantitativo / exploratorio descriptivo / correlacional / explicativo / /estudio de caso) y el tipo de diseño (experimental/no experimental;  transversal/longitudinal; prospectivo/retrospectivo). Se dará cuenta de la unidad/es de análisis y de las principales variables/ejes relevantes a analizar y sus indicadores. Población/muestra.  Unidades de respuesta. Principales técnicas de recolección de datos. Si se va a trabajar con datos secundarios, es necesario mencionar su fuente.  Procedimientos de análisis de los datos.

Se propone (a modo de guía y según corresponda) describir estos puntos en función de cada objetivo específico en una tabla síntesis.

Objetivo específico	Fuente secundaria de datos	Fuente primaria de datos/ Instrumento de recolección	Población/muestra	Técnicas de procesamiento

-->

# Cronograma

| Actividad / Meses | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Revisión Bibliográfica** | X | X | | | | | | | | | | |
| **Fundamentación Teórica (DiD y DML)** | | X | X | | | | | | | | | |
| **Recolección y Limpieza de Datos** | | | X | X | | | | | | | | |
| **Desarrollo Metodológico (Implementación DML)** | | | | X | X | | | | | | | |
| **Análisis de Caso 1 (Fracking)** | | | | | | X | X | | | | | |
| **Análisis de Caso 2 (Castle Doctrine)** | | | | | | | X | X | | | | |
| **Análisis Comparativo y Discusión** | | | | | | | | | X | X | | |
| **Escritura y Redacción Final** | | | | | | | | | | X | X | X |
| **Revisión y Correcciones** | | | | | | | | | | | | X |

<!---
Se presentará un cuadro de doble entrada en el que se especificarán en las filas las actividades a realizar y en las columnas los períodos de tiempo (diagrama de Gantt.).

| Actividad / Meses del año 20XX | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Actividad 1 | | | | | | | | | | | | |
| Actividad 2 | | | | | | | | | | | | |
| Actividad 3 | | | | | | | | | | | | |
| Actividad 4 | | | | | | | | | | | | |
| Actividad 5 | | | | | | | | | | | | |
| Actividad n | | | | | | | | | | | | |



-->


# Referencias bibliográficas y bibliografía (preliminar)

*   **Abadie, A. (2005).** Semiparametric difference-in-differences estimators. *The Review of Economic Studies*.
*   **Breiman, L. (2001).** Statistical Modeling: The Two Cultures. *Statistical Science*.
*   **Callaway, B., & Sant’Anna, P. H. (2021).** Difference-in-differences with multiple time periods. *Journal of Econometrics*.
*   **Chang, N. (2020).** Double/debiased machine learning for difference-in-differences models. *The Econometrics Journal*.
*   **Cheng, C., & Hoekstra, M. (2013).** Does strengthening self-defense law deter crime or escalate violence? Evidence from expansions to castle doctrine. *Journal of Human Resources*.
*   **Chernozhukov, V., et al. (2018).** Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*.
*   **Cunningham, S. (2021).** *Causal Inference: The Mixtape*. Yale University Press.
*   **Gonzales, R. (2025).** *Fracking and Environmental Regulations*.
*   **Goodman-Bacon, A. (2021).** Difference-in-differences with variation in treatment timing. *Journal of Econometrics*.
*   **Varian, H. R. (2014).** Big data and economics. *Journal of Economic Perspectives*.

(Para el listado completo, ver archivo `book/references.bib`)

<!---
Listado de las referencias bibliográficas que se utilizaron en la elaboración del proyecto. Asimismo se sugiere agregar un listado de la bibliografía preliminar (que será ampliada durante el desarrollo del TFM). Se  confeccionará un listado por orden alfabético según apellido del primer autor de la referencia  a listar. Se seguirán las normas APA en su edición más reciente (consultar http://www.apastyle.org/)  
-->
