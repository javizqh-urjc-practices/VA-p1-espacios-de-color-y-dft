[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6bfcAzJo)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13786595&assignment_repo_type=AssignmentRepo)
# Práctica 1 – Espacios de color y DFT

![distro](https://img.shields.io/badge/Ubuntu%2022-Jammy%20Jellyfish-green)
![distro](https://img.shields.io/badge/ROS2-Humble-blue)
[![humble](https://github.com/jmguerreroh/computer_vision/actions/workflows/master.yaml/badge.svg?branch=humble)](https://github.com/jmguerreroh/computer_vision/actions/workflows/master.yaml)

This package is recommended to use with the [TIAGO](https://github.com/jmguerreroh/tiago_simulator) simulator.

# Run

Execute:
```bash
ros2 launch practica1-grupo5 cv.launch.py
```
If you want to use your own robot, in the launcher, change the topic names to match the robot topics.

# Questions

- Adjunta una captura de la opción 3, e indica el por qué se ven esos colores en la
imagen resultado de la resta entre HSV y HSI.

Hay que resumir esto

HSI, HSV, and HSL are all different color spaces. Hue computation is (as far as I can find) identical between the three models, and uses a 6-piece piece-wise function to determine it, or for a simpler model that is accurate to within 1.2 degrees, atan((sqrt(3)⋅(G-B))/2(R-G-B)) can be used. For the most part, these two are interchangeable, but generally HSV and HSL use the piece-wise model, where HSI usually uses the arctan model. Different equations may be used, but these usually sacrifice precision for either simplicity or faster computation.

For lightness/value/intensity, the three spaces use slightly different representations.

Intensity is computed by simply averaging the RGB values: (1/3)⋅(R+G+B).
Value is the simplest, being the value of the maximum of RGB: max(R,G,B).
When used in subsequent calculations, L/V/I is scaled to a decimal between 0 and 1.

Saturation is where the three models differ the most. For all 3, if I/V/L is 0, then saturation is 0 (this is for black, so that its representation is unambiguous), and HSL additionally sets saturation to 0 if lightness is maximum (because for HSL maximum lightness means white).

HSL and HSV account for both the minimum and maximum of RGB, taking the difference between the two: max(R,G,B) - min(R,G,B), this value is sometimes referred to as chroma (C).
HSV then takes the chroma and divides it by the value to get the saturation: C/V.
HSI doesn't use chroma explicitly, instead only taking min(R,G,B) into account: 1 - min(R,G,B)/I.

- ¿Qué se observa en la imagen resultado de la opción 6 y en qué influye que se varíe el
valor del filtro?

## FAQs:

* /usr/bin/ld shows libraries conflicts between two versions:

Probably you have installed and built your own OpenCV version, rename your local folder:
```bash
mv /usr/local/lib/cmake/opencv4 /usr/local/lib/cmake/oldopencv4
```

## About

This is a project made by Javier Izquierdo and Sebastián Andrés Mayorquín, Students at [Universidad Rey Juan Carlos].

Copyright &copy; 2024.

## License

Shield: 

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

[Universidad Rey Juan Carlos]: https://www.urjc.es/
