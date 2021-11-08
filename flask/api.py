from flask import Flask, request
from flask_restx import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)


@api.route('/check', methods=['GET','POST'])
class HelloWorld(Resource):
    def get(self):
        parser=reqparse.RequestParser()
        parser.add_argument('num',type=int,default=65)        
        args=parser.parse_args(strict=True)
        if (args['num']%2)==0:
            return 'Given number is even'
        else:
            return 'Given number is odd'
    def post(self):
        parser=reqparse.RequestParser()
        parser.add_argument('num',type=int,default=65)        
        args=parser.parse_args(strict=True)
        if (args['num']%2)==0:
            return 'Given number is even'
        else:
            return 'Given number is odd'
if __name__ == '__main__':
    app.run()   