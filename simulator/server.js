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

    redisClient.set('machine-state-puck', JSON.stringify(puck));
    pub.publish("state-changed", true);

    res.status(200).send({});
});

app.post('/api/bot-move', (req, res) => {

    bot = req.body;

    redisClient.set('machine-state-bot', JSON.stringify(bot));
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
    const puckRes = redisClient.get("machine-state-puck", function(err, puckState) {
        //io.emit('puck-state-change', reply);
        const botRes = redisClient.get("machine-state-bot", function(err, botState) {

            const state = {
                puck: JSON.parse(puckState),
                bot: JSON.parse(botState)
            };

            publishMachineState(state);
        });
    });
});

function publishMachineState(state) {
    io.emit('state-change', state)
}

sub.subscribe("state-changed");