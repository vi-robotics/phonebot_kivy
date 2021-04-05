#!/usr/bin/env python3
"""
NOTE(ycho-or): This file is necessary in order for pyphonebot
package to be resolved without attempting pip installation.
"""

from pythonforandroid.recipe import PythonRecipe


class PhonebotRecipe(PythonRecipe):
    version = "0.4.0"
    url = 'https://github.com/yycho0108/PhoneBot/archive/0.4.0.tar.gz'
    depends = [ ]
    site_packages_name = 'phonebot'


recipe = PhonebotRecipe()
