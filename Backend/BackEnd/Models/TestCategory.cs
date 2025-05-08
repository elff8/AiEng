using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace BackEnd.Models
{
	public class TestCategory
	{
		public int Id { get; set; }
		public string Name { get; set; }

		public ICollection<Test> Tests { get; set; }
	}



}
