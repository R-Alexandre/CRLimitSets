#!/usr/bin/env python
# -*- coding: utf-8 -*-

GENERATORS = []
RELATIONS = []


def enhance_relations():
    """Cette fonction calcule les relations à rechercher dans les mots.

    La présentation du groupe fondamental fournie par SnapPy ne donne qu'une
    relation. À cela il faut ajouter ses symétriques (inverses et commutations)
    ainsi que les relations évidentes de type aA et Aa.
    """

    # ajoute les mots de la forme aA
    for letter in GENERATORS:
        relation = letter + letter.swapcase()
        RELATIONS.append(relation)
        RELATIONS.append(relation.swapcase())

    # chaque mot peut etre permute circulairement
    # g_1 g_2 = e iff g_2 = G_1 iff g_2 g_1 = e
    for relation in list(RELATIONS):
        for i in range(1,len(relation)):
            x = relation[:i]
            y = relation[i:]
            if (y+x) not in RELATIONS:
                RELATIONS.append(y+x)

    # inverse chaque mot
    for relation in list(RELATIONS):
        new_relation = ''
        for i in range(len(relation)):
            letter = relation[-(i+1)]
            new_relation += letter.swapcase()
        if new_relation not in RELATIONS:
            RELATIONS.append(new_relation)

    return 0

def enhance_generators():

    for generator in list(GENERATORS):
        GENERATORS.append(generator.upper())
    return 0

# Verifie que le mot ne contient pas de partie qui soit dans les relations
def no_relation_contained(word):

    for relation in RELATIONS:
        if relation in word:
            return False

    return True


def lists_forming_words_length(n):
    """ Cette fonction constuit deux listes permettant de constituer tous
    les mots de longueur n.

    Le choix de faire intervenir deux listes plutôt qu'une permet d'économiser
    quadratiquement de la mémoire.

    La construction de chaque liste se fait de façon analogue et par
    récursivité. À chaque étape, on rajoute une lettre et on vérifie que
    le mot obtenu ne contient pas de relation.
    """

    if n%2 == 0:
        list = comprehensive_list_words_length(n/2)
        return (list,list)

    if n%2 == 1:
        list_a = comprehensive_list_words_length((n+1)/2)
        list_b = comprehensive_list_words_length((n-1)/2)
        return (list_a,list_b)

def comprehensive_list_words_length(n):

    if n == 0:
        return ['']

    if n == 1:
        return GENERATORS[:]

    if n%2  == 0:
        intermed_list = comprehensive_list_words_length(n/2)

        list = []
        for a in intermed_list:
            for b in intermed_list:
                if a != b:
                    word = a+b
                    if no_relation_contained(word):
                        list.append(word)
        return list

    if n%2 == 1:
        intermed_list_a = comprehensive_list_words_length((n+1)/2)
        intermed_list_b = comprehensive_list_words_length((n-1)/2)

        list = []
        for a in intermed_list_a:
            for b in intermed_list_b:
                word = a+b
                if no_relation_contained(word):
                    list.append(word)
        return list
