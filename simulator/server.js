const bodyParser = require('body-parser');
const express = require('express');
const app = express();
const socketio = require('socket.io');
var io = {};

const redis = require('redis');
const redisClient = redis.createClient();
const sub = redis.createClient(), pub = redis.createClient();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.post('/api/puck-move', (req, res) => {

    puck = req.body;

    let state = {
        puck: {
            x: puck.x,
            y: puck.y
        },
        bot: {
            x:0,
            y:0
        }
    };

    redisClient.set('machine-state', JSON.stringify(state));
    pub.publish("state-changed", true);

    res.status(200).send({});
});

app.use(express.static('public'));
var server = app.listen(3000, () => console.log('Listening on port 3000'));

// ############## Socket.io stuff ##############
io = socketio.listen(server);

io.on('connection', function(socket) {
    console.log('Client Connected');
});

sub.on("message", function(channel, message) {
    //console.log("sub channel " + channel + ": " + message);
    const state = redisClient.get("machine-state", function(err, reply) {
        console.log(reply.toString());

        io.emit('state-change', reply);
    });
});
sub.subscribe("state-changed");

