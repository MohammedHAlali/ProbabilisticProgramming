Pkg.update()
Pkg.add("SQLite")

using SQLite

db = SQLite.DB("Chinook/Chinook_Sqlite.sqlite")

SQLite.query(db, 
"SELECT FirstName, LastName FROM Employee WHERE LastName REGEXP 'e(?=a)'")

SQLite.query(db, "SELECT * FROM Genre WHERE regexp('e[trs]', Name)")

SQLite.query(db, "SELECT * FROM Genre ORDER BY GenreId LIMIT 2")

dat = SQLite.query(db, "SELECT * FROM Genre ORDER BY GenreId LIMIT 20")

convert( Matrix, dat )
