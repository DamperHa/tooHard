import my_package_tutorial

# Accesses the package_variable from my_package_tutorial
print(my_package_tutorial.package_variable)

# Calls the foo function from module1
my_package_tutorial.module1.foo()

# Importing submodule will trigger the print statement in submodule/__init__.py
from my_package_tutorial import submodule

# Accesses the submodule_variable
print(submodule.submodule_variable)

# Calls the bar function from module2
submodule.module2.bar()
