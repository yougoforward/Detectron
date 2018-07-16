import compileall
compileall.compile_dir(目录)



python -m py_compile /path/to/需要生成.pyc的脚本.py #若批量处理.py文件
                                                  #则替换为/path/to/{需要生成.pyc的脚本1,脚本2,...}.py
                                                  #或者/path/to/

import py_compile
py_compile.compile(r'/path/to/需要生成.pyc的脚本.py') #同样也可以是包含.py文件的目录路径
                                                    #此处尽可能使用raw字符串，从而避免转义的麻烦。比如，这里不加“r”的话，你就得对斜杠进行转义

另外，生成.pyo文件的格式调用如下：
python -O -m py_compile /path/to/需要生成.pyo的脚本.py


若想优化生成字节码，应注意这两点：
.pyc文件是由.py文件经过编译后生成的字节码文件，其加载速度相对于之前的.py文件有所提高，
而且还可以实现源码隐藏，以及一定程度上的反编译。比如，Python3.3编译生成的.pyc文件，
Python3.4就别想着去运行啦！→_→.pyo文件也是优化（注意这两个字，便于后续的理解）编译后的程序（相比于.pyc文件更小），也可以提高加载速度。
但对于嵌入式系统，它可将所需模块编译成.pyo文件以减少容量。

在所有的Python选项中：
-O，表示优化生成.pyo字节码（这里又有“优化”两个字，得注意啦！）
-OO，表示进一步移除-O选项生成的字节码文件中的文档字符串（这是在作用效果上解释的，而不是说从-O选项得到的文件去除）
-m，表示导入并运行指定的模块