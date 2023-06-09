# Kognitív robotika házi feladat

A feladatunk célja:
A turtlebot felismerje a szakadást a vonalon és egy jelzést tegyen oda előszöri lefutáskor, majd a második lefutásnál változtassuk meg egy szakadás helyét és frissítse ezt a robot a térképen.

## A lépések:
Először megtanítottuk a neurális hálónak, hogy felismerje, ha egy szakadást lát.

### A neurális hálónk tanításának az eredménye:
![model_training](https://github.com/pixelb1rd/project/assets/130582814/13606396-2dc7-43f2-9433-bd0ffb510d82)



Majd elkészítettük programkód kiegészítését, ami lehelyez egy jelzést, ha szakadást érzékel.
Itt abba a hibába futottunk, hogy a tanítás során olyan helyeken is szakadást érzékel, ahol nincsen.
Emiatt folyamatosan jelöléseket tesz le, ezért ennél tovább nem tudtunk jutni a feladatunkba.

## A vonalkövetésről készült videónk:
https://github.com/pixelb1rd/project/assets/130582814/8fb592ce-811f-4a38-89e5-9f9651ca69a0



