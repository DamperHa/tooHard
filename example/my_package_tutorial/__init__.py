# This will control what gets imported when you do 'from my_package_tutorial import *'
__all__ = ['module1', 'submodule']

print("Initializing my_package_tutorial")

# You can initialize some package-level variables or settings here
package_variable = "I am a package variable"

# Importing a module from this package for convenience for the user
from . import module1
