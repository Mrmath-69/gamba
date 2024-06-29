
async function nflGames(){
    const req = await fetch("http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard")
    const data = await req.json()
    var events = data.events;

    events.forEach(event => {
        console.log(`Event Date: ${event.date}`);
        console.log(`Matchup: ${event.shortName}`);
        console.log(`Week Number: ${event.week.number}`);
        console.log("-----------------------");
    });

}
nflGames();

