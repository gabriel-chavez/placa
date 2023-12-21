import string
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}
def license_complies_format(text):
    """
    Comprueba si el texto de la matrícula cumple con el formato requerido.

    Args:
        text (str): Texto de la matrícula.

    Returns:
        bool: True si la matrícula cumple con el formato, False en caso contrario.
    """
    if len(text)==6:
        if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
        (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
        (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()):
            return True
        else:
            return False
        
    if len(text)==7:
        if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
        (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
            return True
        else:
            return False
   
    return False

# Prueba con la placa "123ABC" (debería ser True)
placa_valida = license_complies_format("A23ABC")
print(placa_valida)

# Prueba con la placa "12ABC" (debería ser False)
placa_valida = license_complies_format("12ABC")
print(placa_valida)

# Prueba con la placa "ABC123" (debería ser False)
placa_valida = license_complies_format("ABC123")
print(placa_valida)

# Prueba con la placa "1234ABC" (debería ser False)
placa_valida = license_complies_format("123AABC")
print(placa_valida)
