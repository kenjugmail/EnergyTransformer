# Fixed-ID output examples

These examples use the first declared prompt and first six task identities per variant and seed; they are not selected for correctness.

**Transformer / seed 101**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
un alle on te thed as de mof amast mponve (cerontheel whed tintestin by se alof Fr us s thal. f 
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=11,21,319,21144,34,31444;ans=1

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=S,S;ans=S

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=1;ans=1

```
Expected: `steps=1;ans=1`; complete correctness: True.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=19,31;ans=31

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=I,A,A;ans=U

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=1;ans=7

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Raw REWT control / seed 101**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
ura]], [[Hate]] ([[1952]]).

===Cognor xacal molutane with time to chateor and [[nation pree non
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=16,21,33;ans=31

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=R,T;ans=T

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=8;ans=8

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=18,38;ans=8

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=Q,Q,Q;ans=Q

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=0;ans=0

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Revised neural energy / seed 101**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
y a commines doogh and the bean sublouguia consive of which the to ematemens pie this of the sig
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=28,30,40;ans=40

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=C,C;ans=C

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=4;ans=4

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=28,34;ans=44

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=Q;ans=Q

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=4;ans=4

```
Expected: `steps=4;ans=4`; complete correctness: True.

**Revised Fourier REWT / seed 101**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
y a conkines and in the dither dit othes [[Firthy]] daye]]
[[et:roum]], [[Cantem]]
|-
| on [[gla
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=22,32,38;ans=38

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=O,F;ans=F

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=8;ans=8

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=18,38;ans=38

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=Q;ans=Q

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=9;ans=9

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Revised full REWT / seed 101**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
y and heap the denalle. Ans ale sublopey [[Firthy Forb|[[Lastater]] [[seminais]]
[[st:NA�]]
*[[F
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=20,30,40;ans=40

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=Q,C,W;ans=C

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=3;ans=3

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=18,28;ans=28

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=C,Q,A;ans=A

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=7;ans=7

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Transformer / seed 202**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
 d hetentpoth bontre m ct thon spar sh tithican th. coustres alall athampe whe thithatore [[Jome
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=8,17,40,17,40,4;ans=26

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=S,S;ans=S

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=7;ans=7

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=24,34;ans=34

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=S,S;ans=S

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=15,19,37;ans=3

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Raw REWT control / seed 202**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
e beet systrican to dices the [[Scolooding]]
* [http://www.thb///wwww.clugaunt/retum/tound.7%000
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=19,33,49;ans=39

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=K,W;ans=W

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=7;ans=7

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=19,29;ans=29

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=K,L,L;ans=L

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=7;ans=7

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Revised neural energy / seed 202**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
e converts the mays commuming and the Grown.

==Shecemspornon and a the thes siters by the emper
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=12,28,33;ans=33

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=I,I;ans=I

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=0;ans=0

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=16,25;ans=15

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=Q,Q;ans=Q

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=0;ans=0

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Revised Fourier REWT / seed 202**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
e convertrounceinnon of the the parch as the Spate forpory the [[Schymall Cunter sugeration comp
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=21,31,33;ans=31

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=V,P;ans=P

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=0;ans=0

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=21,21;ans=21

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=J,Q,Q;ans=J

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=0;ans=0

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Revised full REWT / seed 202**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
e bely orlpriccints bo dubsiticllas of the scanctice, with the beraks, by ater the devison in [[
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=16,28,36;ans=36

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=R,I,I;ans=I

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=2;ans=2

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=11,28;ans=29

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=S,U,D;ans=U

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=2;ans=2

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Transformer / seed 303**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
 s anditsiche rore me bithererely pre g ised af The arvitheey po io ts me t me ban s thitonin f 
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=21,24,4;ans=441

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=M,M;ans=M

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=,;ans=40

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=24,34;ans=34

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=C,C,C;ans=C

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=4;ans=44

```
Expected: `steps=4;ans=4`; complete correctness: False.

**Raw REWT control / seed 303**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
ation to sell reshere devillation of [[1986]]
* Book its anderded the [[Sin]] Alandicancy not ch
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=20,29,39;ans=39

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=N,F;ans=F

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=4;ans=4

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=20,29;ans=20

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=J,Q;ans=J

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=4;ans=4

```
Expected: `steps=4;ans=4`; complete correctness: True.

**Revised neural energy / seed 303**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
den incturance]], [[Indes]]--chited has his a agate and the was alogowan, the chanch the soth ee
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=19,20,20;ans=20

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=X,Q,T;ans=Q

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=4;ans=4

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=20,20;ans=20

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=J,Q,Q;ans=Q

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=4;ans=4

```
Expected: `steps=4;ans=4`; complete correctness: True.

**Revised Fourier REWT / seed 303**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
ation to scleation legely his of etted and mage and [[tromaty meand or of the chanch the sour-ce
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=19,29,32;ans=29

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=X,T;ans=T

```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=4;ans=4

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=27,30;ans=30

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=J,J;ans=J

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=4;ans=4

```
Expected: `steps=4;ans=4`; complete correctness: True.

**Revised full REWT / seed 303**

Prompt:

```text
 forbid that which Allah has mad
```

Continuation:

```text
]] - [[purand ished he frome, promes an, since befed trith ox stane to mens manad, shothis sode 
```

Task e2df1d1c101b (arithmetic):

```text
A=11;B=A+8;C=B+5;D=C+12;D=?
steps=2,3,40;ans=40

```
Expected: `steps=19,24,36;ans=36`; complete correctness: False.

Task fb102959772b (rules):

```text
S>F;R>Q;F>G;Q>M;from S go 2=?
steps=M,T,20,30,30,30,30,30,30,30,30,30,
```
Expected: `steps=F,G;ans=G`; complete correctness: False.

Task 65003d74dc97 (recall):

```text
key 1;X=5;X=1;X=3;X=5;X=7;X=9;X=7;X=8;key?
steps=3;ans=3

```
Expected: `steps=1;ans=1`; complete correctness: False.

Task 0865c9a2fec1 (arithmetic):

```text
A=16;B=A+11;C=B+5;C=?
steps=26,34,46;ans=46

```
Expected: `steps=27,32;ans=32`; complete correctness: False.

Task 5505392a1707 (rules):

```text
walk P 3:P>B;Q>V;J>C;B>J;L>Q=?
steps=J,J;ans=J

```
Expected: `steps=B,J,C;ans=C`; complete correctness: False.

Task 20c72f61c8bd (recall):

```text
key 4;X=6;X=5;X=7;X=0;key?
steps=1;ans=1

```
Expected: `steps=4;ans=4`; complete correctness: False.
