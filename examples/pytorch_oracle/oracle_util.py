from functools import reduce

def save_oracle_result(case_name, oracle_result):

    result_hpp = '''

#ifndef HYPERTEA_TEST_{0}_RESULT_HPP_
#define HYPERTEA_TEST_{0}_RESULT_HPP_

#include <vector>

namespace hypertea {{

namespace test_result {{

{1}

}} //namespace test_result

}} //namespace hypertea

#endif //HYPERTEA_TEST_{0}_RESULT_HPP_
        
    '''.format(case_name.upper(), ('\n'*4).join(oracle_result))


    with open('oracle_hpp/{}_result.hpp'.format(case_name), 'w') as f:

        f.write(result_hpp)


def list2vecstr_(l, vec_type = 'int'):
        return 'std::vector<{}> {{'.format(vec_type)  + ','.join(map(str, l)) + '}'

def bool2str_(x):
    return 'true' if x else 'false'

def bool2inplace_str_(x):
    return 'IN_PLACE' if x else 'NOT_IN_PLACE'


def prod_(l):
    return reduce(lambda x, y: x*y, l)