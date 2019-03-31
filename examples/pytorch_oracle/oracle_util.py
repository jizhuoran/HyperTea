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