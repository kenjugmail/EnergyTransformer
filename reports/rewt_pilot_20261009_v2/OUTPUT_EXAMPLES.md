# Fixed generated-output examples

These are the first six test-task IDs and the first text prefix from the frozen plan, seed 101. They are shown regardless of whether they are correct. Full outputs for every case remain in the JSONL files.

## arithmetic: 8c1dc3e56ab9

Prompt:
~~~text
sum 7+6+9=?
~~~
Correct response:
~~~text
steps=13,22;ans=22
~~~

Transformer (answer correct: False; complete correct steps: False):
~~~text
steps=1;ans=14
~~~

Neural energy (answer correct: False; complete correct steps: False):
~~~text
steps=9,12;ans=20
~~~

Fourier REWT (answer correct: False; complete correct steps: False):
~~~text
steps=E,14;ans=1
~~~

Full REWT (answer correct: False; complete correct steps: False):
~~~text
steps=1;ans=1
~~~

## rules: 4a0e4645e283

Prompt:
~~~text
V>D;J>V;N>C;C>U;E>J;X>E;from X go 4=?
~~~
Correct response:
~~~text
steps=E,J,V,D;ans=D
~~~

Transformer (answer correct: True; complete correct steps: False):
~~~text
steps=D;ans=D
~~~

Neural energy (answer correct: False; complete correct steps: False):
~~~text
steps=B,K;ans=T
~~~

Fourier REWT (answer correct: False; complete correct steps: False):
~~~text
steps=E,C,C;ans=1
~~~

Full REWT (answer correct: False; complete correct steps: False):
~~~text
steps=E,F;ans=F
~~~

## recall: b5e0981ce037

Prompt:
~~~text
key 0;X=6;X=3;X=6;X=3;X=3;X=6;X=1;X=4;X=3;X=1;X=4;X=8;X=7;X=2;X=7;X=8;key?
~~~
Correct response:
~~~text
steps=0;ans=0
~~~

Transformer (answer correct: False; complete correct steps: False):
~~~text
steps=1;ans=1;ans=10;ans=3
~~~

Neural energy (answer correct: False; complete correct steps: False):
~~~text
steps=1;ans=6
~~~

Fourier REWT (answer correct: False; complete correct steps: False):
~~~text
steps=1;ans=1
~~~

Full REWT (answer correct: False; complete correct steps: False):
~~~text
steps=1;ans=1
~~~

## arithmetic: 044c57ede5a1

Prompt:
~~~text
start 2;add 4;add 5;result?
~~~
Correct response:
~~~text
steps=6,11;ans=11
~~~

Transformer (answer correct: False; complete correct steps: False):
~~~text
steps=1,1,1;ans=13
~~~

Neural energy (answer correct: False; complete correct steps: False):
~~~text
steps=12,15;ans=22
~~~

Fourier REWT (answer correct: False; complete correct steps: False):
~~~text
steps=1,11,17;ans=12
~~~

Full REWT (answer correct: False; complete correct steps: False):
~~~text
steps=10,17,17;ans=17
~~~

## rules: bf65fabedbe9

Prompt:
~~~text
map:N>K;T>S;S>F;N after 1 links?
~~~
Correct response:
~~~text
steps=K;ans=K
~~~

Transformer (answer correct: False; complete correct steps: False):
~~~text
s=D;ans=D;ans=V
~~~

Neural energy (answer correct: False; complete correct steps: False):
~~~text
steps=B,Q;ans=T
~~~

Fourier REWT (answer correct: False; complete correct steps: False):
~~~text
steps=E,C,14;ans=14
~~~

Full REWT (answer correct: False; complete correct steps: False):
~~~text
steps=E,C,C;ans=G
~~~

## recall: 191b8823429b

Prompt:
~~~text
K=5;X=7;K=?
~~~
Correct response:
~~~text
steps=5;ans=5
~~~

Transformer (answer correct: False; complete correct steps: False):
~~~text
steps=1;ans=1
~~~

Neural energy (answer correct: False; complete correct steps: False):
~~~text
steps=9;ans=9
~~~

Fourier REWT (answer correct: False; complete correct steps: False):
~~~text
steps=1;ans=1
~~~

Full REWT (answer correct: True; complete correct steps: True):
~~~text
steps=5;ans=5
~~~

## First fixed text prefix

~~~text
insurance carriers based on the 
~~~

Transformer:
~~~text
thcals ' amaprmeod a in aoest  ctowiseg&s. n ky t s is s wheneFave kfsan acive san thedeanthare 
~~~

Neural energy:
~~~text
the by A chantinna c de coctrd atowlore the Antes riop reurend ate nound alive semas. Somiticor 
~~~

Fourier REWT:
~~~text
the as      asedec a in an tof by wisincted Dives rion s<upgine we nound alive semas. Somiticor 
~~~

Full REWT:
~~~text
the as ' amelsinii], chalicot  atly opeathe  funtased]. Rthe ceave inteace ly  ond the particne 
~~~

