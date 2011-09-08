use inc::Module::Install;
name 'Algorithm-AdaBoost';
all_from 'lib/Algorithm/AdaBoost.pm';

requires 'Carp';
requires 'Class::Accessor::Lite';
requires 'Exporter::Lite';

tests 't/*.t';
author_tests 'xt';

build_requires 'Test::More';
use_test_base;
auto_include;
WriteAll;
