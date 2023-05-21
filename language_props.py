word_order_to_idx = {
    "None": 0,
    "SVO":  1,
    "SOV":  2,
}
writing_system_to_idx = {
    "alphabetic" :  0,
    "consont" :     1,
    "devanagri" :   2,
}

lang_to_typology_dict = {
    'en': {"word_order": word_order_to_idx["None"], "writing_system": writing_system_to_idx["alphabetic"]}, 
    'nl': {"word_order": word_order_to_idx["None"], "writing_system": writing_system_to_idx["alphabetic"]}, 
    'fy': {"word_order": word_order_to_idx["None"], "writing_system": writing_system_to_idx["alphabetic"]}, 
    'he': {"word_order": word_order_to_idx["SVO"], "writing_system": writing_system_to_idx["consont"]}, 
    'ar': {"word_order": word_order_to_idx["SVO"], "writing_system": writing_system_to_idx["consont"]}, 
    'hi': {"word_order": word_order_to_idx["SOV"], "writing_system": writing_system_to_idx["devanagri"]}, 
    'ur': {"word_order": word_order_to_idx["SOV"], "writing_system": writing_system_to_idx["devanagri"]}
}

