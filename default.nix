{ pkgs ? import <nixpkgs> {} }:

pkgs.python3115.withPackages (ps: with ps; [
    fastapi
    hypercorn
    keras
    tensorflow
    numpy
    Pillow
    requests
])
