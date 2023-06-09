# Kognitív robotika házi feladat

A feladatunk célja:
A turtlebot felismerje a szakadást a vonalon és egy jelzést tegyen oda előszöri lefutáskor, majd a második lefutásnál változtassuk meg egy szakadás helyét és frissítse ezt a robot a térképen.

## A lépések:
Először megtanítottuk a neurális hálónak, hogy felismerje, ha egy szakadást lát.

### A neurális hálónk tanításának az eredménye:
![model_training](https://github.com/pixelb1rd/project/assets/130582814/fe4a2927-ce4d-4e6c-89d0-3751a068b869)


Majd elkészítettük programkód kiegészítését, ami lehelyez egy jelzést, ha szakadást érzékel.
Itt abba a hibába futottunk, hogy a tanítás során olyan helyeken is szakadást érzékel, ahol nincsen.
Emiatt folyamatosan jelöléseket tesz le.
