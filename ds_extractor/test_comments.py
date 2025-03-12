import unittest

from ds_extractor.comments import CommentPreprocessor


class CommentPreprocessorTest(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.under_test = CommentPreprocessor(tokens_replacement=[('ph_user', ['nn', 'sozi ahmad','jack jone']),
                                                                  ('ph_name', ['cc', 'bb'])])

    def test_simple_case_1(self):
        self.__do_test(['greetings', 'is it confirmed by your security team ?'],
                       """
        <p>Dear Faiza , </p><p>Is it confirmed by your security team ?</p>
        """)

    def test_simple_case_2(self):
        self.__do_test(["dear yazan",
                        "all required files uploaded ( iam folders and ph_file for all ph_user system from production ) ."],
                       """
                           <p>Dear Yazan</p><p><br />all required files uploaded ( IAM folders and servr.xml for all NN system from production ) . </p>        
                           """)

    def test_complex_1(self):
        self.__do_test(['greetings',
                        'this ticket is created on the behalf of ph_name bank to track our activities on the issue: query from ph_name bank the check they scan is clear but when we receive is not clear meaning clarity of the image is not quite clear.'],
                       """
                       <p>Dear <span>&lt;~ae.citi.02&gt;</span>,</p>
                       <p><span style="background-color:rgb(255,255,255);color:rgb(0,0,0)">This ticket is created on the behalf of CC bank to track our activities on the issue: </span><span style="background-color:rgb(255,255,255);color:rgb(32,31,30)">Query from BB Bank the check they scan is Clear but when we receive is not Clear  meaning Clarity of the image is not quite clear.</span></p>
                       """)

    def test_complex_2(self):
        self.__do_test(['greetings',
                        'a video was uploaded showing the issue where the user will log out after getting a success qr scan.',
                        'video : ph_file',
                        'regards,',
                        'ph_user'],
                       """
                       <p>Dear <span>[~u1888]</span> ,</p><p>A video was uploaded showing the issue where the user will log out after getting a success QR scan.</p><p>Video :<strong> </strong><span style="background-color:rgb(255,255,255);color:rgb(83,82,82)"><strong>IMG_9762.MP4</strong></span></p><p>Regards,</p><p>Sozi ahmad</p>
                       """)

    def test_simple_case_3(self):
        self.__do_test([
            'vgreetings',
            'kindly if there any update form the development team please provide it.',
            'regards, ph_user',
            'ph_user ph_user'
        ],
            """
            <p>vDear <span>[~u806]</span> </p><p>kindly if there any update form the development team please provide it.</p><p>Regards,<br />jack jone</p><p><span>[~JOSD_user]</span> <br /><span>[~CheckOPS4]</span> </p>
            """)

    def __do_test(self, expected, text):
        actual = self.under_test.preprocess_comment(text)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
